#!/usr/bin/env python3
"""
Lane Follower Node  (Merged-Cloud Edition)
==========================================
Drives the robot forward by computing the lane centre from a single
merged PointCloud2 published by road_detector_node on
/road_detector/pointcloud.

Instead of expecting pre-split left/right clouds, this node:
  1. Reads all road-feature points (lane markings + potholes) from the
     merged cloud.
  2. At each depth slice along Z, splits points into left (X > 0) and
     right (X < 0) groups using a simple lateral sign heuristic.
  3. Computes the lane centre as (median_left_x + median_right_x) / 2,
     falling back to median_x of all points when only one side is visible.
  4. Sends NavigateToPose goals to Nav2 so that obstacle avoidance (via
     the local costmap) is handled automatically.

Key design points
-----------------
  * Intentional-cancel flag prevents genuine Nav2 failures from being
    confused with goal preemption when a new goal is sent.
  * Goal dedup: a new goal is sent only when the lane centre has shifted
    more than `goal_update_dist` metres OR the previous goal is done.
  * Curve detection: when the computed centre is laterally offset beyond
    `curve_detect_thresh`, lookahead is shortened to `curve_lookahead`.
  * Blind fallback: if no road points are visible, the robot coasts
    straight ahead at a short distance to avoid stopping in place.
"""

import math
import uuid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
import tf2_ros
import tf2_geometry_msgs          # noqa: F401 — registers TF2 geometry transforms
from tf2_ros import TransformException
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


def _quat_from_yaw(yaw: float):
    """Return (w, x, y, z) quaternion for a pure yaw rotation."""
    half = yaw * 0.5
    return math.cos(half), 0.0, 0.0, math.sin(half)


class LaneFollowerNode(Node):

    def __init__(self):
        super().__init__('lane_follower_node')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('lookahead_distance',  1.0)   # metres — shorter than original (robot scale)
        self.declare_parameter('lookahead_step',      0.25)  # depth resolution per slice
        self.declare_parameter('slice_tolerance',     0.15)  # ± Z tolerance per slice
        self.declare_parameter('min_remaining_dist',  0.4)   # re-plan when this close to goal
        self.declare_parameter('startup_delay_sec',   0.0)
        self.declare_parameter('nav_goal_timeout',   20.0)
        self.declare_parameter('abort_cooldown_sec',  2.0)
        self.declare_parameter('lane_width',          0.7)   # expected lane width (m) for fallback
        self.declare_parameter('curve_detect_thresh', 0.06)  # |cx| threshold to flag a curve
        self.declare_parameter('curve_lookahead',     0.3)   # shortened lookahead on curves
        self.declare_parameter('goal_update_dist',    0.20)  # min shift to trigger re-plan
        self.declare_parameter('min_side_points',     3)     # min points to trust a side estimate
        # Topic
        self.declare_parameter('cloud_topic', '/road_detector/pointcloud')

        self._lookahead      = self.get_parameter('lookahead_distance').value
        self._step           = self.get_parameter('lookahead_step').value
        self._z_tol          = self.get_parameter('slice_tolerance').value
        self._min_dist       = self.get_parameter('min_remaining_dist').value
        self._startup_sec    = self.get_parameter('startup_delay_sec').value
        self._timeout_sec    = self.get_parameter('nav_goal_timeout').value
        self._abort_cooldown = self.get_parameter('abort_cooldown_sec').value
        self._lane_width     = self.get_parameter('lane_width').value
        self._curve_thresh   = self.get_parameter('curve_detect_thresh').value
        self._curve_la       = self.get_parameter('curve_lookahead').value
        self._update_dist    = self.get_parameter('goal_update_dist').value
        self._min_side_pts   = self.get_parameter('min_side_points').value
        self._cloud_topic    = self.get_parameter('cloud_topic').value

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        # ------------------------------------------------------------------
        # Nav2 action client
        # ------------------------------------------------------------------
        self._nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # ------------------------------------------------------------------
        # Road-feature point cloud  (merged — single topic)
        # ------------------------------------------------------------------
        self._road_pts = np.empty((0, 3), dtype=np.float32)
        self.create_subscription(
            PointCloud2, self._cloud_topic, self._cloud_cb, 5)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self._goal_handle        = None
        self._goal_active        = False
        self._current_goal       = None
        self._goal_sent_at       = None
        self._last_abort_at      = None
        self._started            = False
        self._start_wall         = self.get_clock().now()
        self._intentional_cancel = False
        self._pending_goal_id    = None

        # ------------------------------------------------------------------
        # 0.5 s main loop
        # ------------------------------------------------------------------
        self._timer = self.create_timer(0.5, self._loop)

        self.get_logger().info(
            f'LaneFollowerNode ready  cloud=[{self._cloud_topic}]  '
            f'lookahead={self._lookahead}m  lane_width={self._lane_width}m')

    # ------------------------------------------------------------------
    # Point-cloud callback
    # ------------------------------------------------------------------
    def _cloud_cb(self, msg: PointCloud2):
        """Unpack the merged road-feature cloud."""
        pts = list(point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True))
        if pts:
            self._road_pts = np.array(
                [[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
        else:
            self._road_pts = np.empty((0, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _loop(self):
        # Startup delay
        if self._startup_sec > 0 and not self._started:
            elapsed = (self.get_clock().now() - self._start_wall).nanoseconds / 1e9
            if elapsed < self._startup_sec:
                return

        if not self._started:
            self._started = True
            self.get_logger().info('Lane follower: starting dynamic tracking')

        # Abort cooldown after genuine Nav2 failure
        if self._last_abort_at is not None:
            since = (self.get_clock().now() - self._last_abort_at).nanoseconds / 1e9
            if since < self._abort_cooldown:
                return

        tf = self._robot_tf()
        if tf is None:
            return

        # Goal timeout guard
        if self._goal_active and self._goal_sent_at is not None:
            age = (self.get_clock().now() - self._goal_sent_at).nanoseconds / 1e9
            if age > self._timeout_sec:
                self.get_logger().warn('Goal timed out — cancelling')
                self._cancel_intentionally()

        if not self._goal_active or self._close_to_goal(tf):
            goal = self._compute_goal(tf)
            if goal:
                self._send_goal(goal)
        else:
            goal = self._compute_goal(tf)
            if goal and self._shifted_enough(goal):
                self._send_goal(goal)

    # ------------------------------------------------------------------
    # Goal computation — merged cloud, internal left/right split
    # ------------------------------------------------------------------
    def _compute_goal(self, tf) -> PoseStamped:
        """
        Compute a Nav2 goal from the merged road-feature point cloud.

        Strategy:
          For each depth slice (Z = step, 2*step, … lookahead):
            - Points with X > 0  → left wall candidates
            - Points with X < 0  → right wall candidates
          Lane centre = (median_left_x + median_right_x) / 2
          If only one side visible, offset by ±lane_width/2 from that side.
          If curve detected, shorten lookahead.
        """
        slices   = np.arange(self._step,
                             self._lookahead + self._step * 0.5,
                             self._step)
        best_z   = None
        best_cx  = None
        on_curve = False

        for z in slices:
            lx, rx = self._lane_centre_at(z)

            if lx is not None and rx is not None:
                cx = (lx + rx) / 2.0
                if best_z is None and abs(cx) > self._curve_thresh:
                    on_curve = True
                best_cx = cx
                best_z  = z

            elif lx is not None or rx is not None:
                if best_z is None:          # only record nearest partial slice
                    best_cx = (lx + self._lane_width / 2.0) if lx is not None \
                         else (rx - self._lane_width / 2.0)
                    best_z  = z

        # On a curve use shorter, safer lookahead
        if on_curve and best_z is not None:
            best_z  = min(best_z, self._curve_la)
            lx, rx  = self._lane_centre_at(best_z)
            if lx is not None and rx is not None:
                best_cx = (lx + rx) / 2.0
            elif lx is not None:
                best_cx = lx + self._lane_width / 2.0
            elif rx is not None:
                best_cx = rx - self._lane_width / 2.0

        if best_z is None or best_cx is None:
            self.get_logger().warn(
                'No road points visible — blind fallback',
                throttle_duration_sec=3.0)
            return self._blind_goal(tf)

        # Clamp lateral offset to 60 % of lane half-width
        best_cx = float(np.clip(best_cx,
                                -self._lane_width * 0.6,
                                 self._lane_width * 0.6))

        # Build point in camera_link_optical frame → transform to map
        pt_cam                  = PointStamped()
        pt_cam.header.frame_id  = 'camera_link_optical'
        pt_cam.header.stamp     = rclpy.time.Time().to_msg()
        pt_cam.point.x          = float(best_cx)
        pt_cam.point.y          = 0.0
        pt_cam.point.z          = float(best_z)

        try:
            pt_map = self._tf_buf.transform(
                pt_cam, 'map', timeout=Duration(seconds=0.5))
        except TransformException as e:
            self.get_logger().warn(
                f'TF camera→map failed: {e} — blind fallback',
                throttle_duration_sec=3.0)
            return self._blind_goal(tf)

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y
        yaw = math.atan2(pt_map.point.y - ry, pt_map.point.x - rx)
        w, qx, qy, qz = _quat_from_yaw(yaw)

        pose                      = PoseStamped()
        pose.header.frame_id      = 'map'
        pose.header.stamp         = self.get_clock().now().to_msg()
        pose.pose.position.x      = pt_map.point.x
        pose.pose.position.y      = pt_map.point.y
        pose.pose.position.z      = 0.0
        pose.pose.orientation.w   = w
        pose.pose.orientation.x   = qx
        pose.pose.orientation.y   = qy
        pose.pose.orientation.z   = qz
        return pose

    def _lane_centre_at(self, z_target: float):
        """
        Return (left_median_x, right_median_x) at depth z_target.

        Points are split by lateral sign:
          X > 0  → left side  (in camera_link_optical, +X is left)
          X < 0  → right side
        Returns None for a side when fewer than min_side_points are found.
        """
        if self._road_pts.size == 0:
            return None, None

        mask  = ((self._road_pts[:, 2] > z_target - self._z_tol) &
                 (self._road_pts[:, 2] < z_target + self._z_tol))
        slice_pts = self._road_pts[mask]

        if slice_pts.size == 0:
            return None, None

        left  = slice_pts[slice_pts[:, 0] >  0.0]
        right = slice_pts[slice_pts[:, 0] <= 0.0]

        lx = float(np.median(left[:, 0]))  if len(left)  >= self._min_side_pts else None
        rx = float(np.median(right[:, 0])) if len(right) >= self._min_side_pts else None

        # Edge case: all points are on one side — treat as full-lane fallback
        if lx is None and rx is None and len(slice_pts) >= self._min_side_pts:
            cx = float(np.median(slice_pts[:, 0]))
            # Assume it could be either wall; return as "both" with width offset
            if cx > 0:
                lx = cx
            else:
                rx = cx

        return lx, rx

    # ------------------------------------------------------------------
    # Blind forward goal (no visible lane)
    # ------------------------------------------------------------------
    def _blind_goal(self, tf) -> PoseStamped:
        x   = tf.transform.translation.x
        y   = tf.transform.translation.y
        q   = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        d   = self._curve_la * 0.4       # very short blind step

        pose                    = PoseStamped()
        pose.header.frame_id    = 'map'
        pose.header.stamp       = self.get_clock().now().to_msg()
        pose.pose.position.x    = x + d * math.cos(yaw)
        pose.pose.position.y    = y + d * math.sin(yaw)
        pose.pose.position.z    = 0.0
        pose.pose.orientation   = q
        return pose

    # ------------------------------------------------------------------
    # Goal bookkeeping helpers
    # ------------------------------------------------------------------
    def _close_to_goal(self, tf) -> bool:
        if self._current_goal is None:
            return True
        dx = tf.transform.translation.x - self._current_goal.pose.position.x
        dy = tf.transform.translation.y - self._current_goal.pose.position.y
        return math.hypot(dx, dy) < self._min_dist

    def _shifted_enough(self, new_pose: PoseStamped) -> bool:
        if self._current_goal is None:
            return True
        dx = new_pose.pose.position.x - self._current_goal.pose.position.x
        dy = new_pose.pose.position.y - self._current_goal.pose.position.y
        return math.hypot(dx, dy) >= self._update_dist

    def _robot_tf(self):
        try:
            return self._tf_buf.lookup_transform(
                'map', 'base_footprint',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2))
        except TransformException:
            return None

    # ------------------------------------------------------------------
    # Goal sending / callbacks
    # ------------------------------------------------------------------
    def _send_goal(self, pose: PoseStamped):
        if not self._nav.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Nav2 action server not available')
            return

        self._cancel_intentionally()   # preempt any active goal

        goal_msg      = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(
            f'New goal → ({pose.pose.position.x:.2f}, '
            f'{pose.pose.position.y:.2f})')

        self._pending_goal_id = str(uuid.uuid4())
        fut = self._nav.send_goal_async(goal_msg)
        fut.add_done_callback(
            lambda f, gid=self._pending_goal_id: self._on_accepted(f, gid))

        self._goal_active  = True
        self._current_goal = pose
        self._goal_sent_at = self.get_clock().now()

    def _on_accepted(self, future, goal_id: str):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2')
            self._goal_active = False
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda f: self._on_result(f, goal_id))

    def _on_result(self, future, goal_id: str):
        if goal_id != self._pending_goal_id:
            return   # stale result from superseded goal

        if self._intentional_cancel:
            # We cancelled to send a newer goal — not a failure
            self._intentional_cancel = False
            self._goal_active  = False
            self._goal_handle  = None
            return

        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().debug('Goal succeeded')
        elif status == GoalStatus.STATUS_CANCELED:
            pass   # already handled above or externally
        else:
            self.get_logger().warn(
                f'Goal failed (status={status}) — '
                f'cooldown {self._abort_cooldown}s')
            self._last_abort_at = self.get_clock().now()

        self._goal_active = False
        self._goal_handle = None

    def _cancel_intentionally(self):
        """Cancel current goal and flag it as intentional (not a failure)."""
        if self._goal_handle is not None:
            self._intentional_cancel = True
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cancel_intentionally()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()