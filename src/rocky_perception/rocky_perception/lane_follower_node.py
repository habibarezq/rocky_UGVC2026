#!/usr/bin/env python3
"""
Lane Follower Node  (Vision-Aware + Curve-Safe)
================================================
Drives the robot forward by computing the geometric centre between the
left (yellow) and right (white) virtual-wall point clouds.

Fixes in this version
---------------------
  * Preemption vs genuine failure distinguished via _intentional_cancel flag.
    Nav2 always returns STATUS_ABORTED (6) when a goal is preempted by a new
    one — previously this triggered a 2-second cooldown on EVERY goal update.
    Now only genuine Nav2 failures (no progress, planner error) trigger it.

  * Goal update rate reduced to 0.5 s and dedup threshold raised to 0.30 m.
    This drastically reduces preemptions — the robot now follows a goal for
    longer before it is replaced, giving the controller time to actually move.

  * Goal is only replaced when it has moved > 0.30 m OR the previous goal
    succeeded/failed (not just "we've been waiting 0.3 s").

  * Multi-slice centroid + adaptive lookahead on curves unchanged from v2.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5);  sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)
    return [
        cy * cp * cr + sy * sp * sr,   # w
        cy * cp * sr - sy * sp * cr,   # x
        sy * cp * sr + cy * sp * cr,   # y
        sy * cp * cr - cy * sp * sr,   # z
    ]


class LaneFollowerNode(Node):

    def __init__(self):
        super().__init__('lane_follower_node')

        # Parameters -------------------------------------------------------
        self.declare_parameter('lookahead_distance',  1.5)
        self.declare_parameter('lookahead_step',      0.5)
        self.declare_parameter('slice_tolerance',     0.25)
        self.declare_parameter('min_remaining_dist',  0.4)
        self.declare_parameter('startup_delay_sec',   0.0)
        self.declare_parameter('nav_goal_timeout',    30.0)
        self.declare_parameter('abort_cooldown_sec',  2.0)
        self.declare_parameter('lane_width',          0.75)
        self.declare_parameter('curve_detect_thresh', 0.12)
        self.declare_parameter('curve_lookahead',     0.8)
        # Goal is only replaced when it shifts more than this (m)
        self.declare_parameter('goal_update_dist',    0.30)

        self._lookahead       = self.get_parameter('lookahead_distance').value
        self._step            = self.get_parameter('lookahead_step').value
        self._z_tol           = self.get_parameter('slice_tolerance').value
        self._min_dist        = self.get_parameter('min_remaining_dist').value
        self._startup_sec     = self.get_parameter('startup_delay_sec').value
        self._timeout_sec     = self.get_parameter('nav_goal_timeout').value
        self._abort_cooldown  = self.get_parameter('abort_cooldown_sec').value
        self._lane_width      = self.get_parameter('lane_width').value
        self._curve_thresh    = self.get_parameter('curve_detect_thresh').value
        self._curve_lookahead = self.get_parameter('curve_lookahead').value
        self._update_dist     = self.get_parameter('goal_update_dist').value

        # TF
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        # Nav2 action client
        self._nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # Vision subscriptions
        self._left_pts  = np.empty((0, 3), dtype=np.float32)
        self._right_pts = np.empty((0, 3), dtype=np.float32)
        self.create_subscription(PointCloud2, '/virtual_walls/left',  self._left_cb,  5)
        self.create_subscription(PointCloud2, '/virtual_walls/right', self._right_cb, 5)

        # State
        self._goal_handle       = None
        self._goal_active       = False
        self._current_goal      = None
        self._goal_sent_at      = None
        self._last_abort_at     = None
        self._started           = False
        self._start_wall        = self.get_clock().now()

        # ---------------------------------------------------------------
        # KEY FIX: track intentional cancellations so preempted goals
        # don't trigger the abort cooldown.
        # ---------------------------------------------------------------
        self._intentional_cancel = False

        # 0.5 s loop — slower than v2's 0.3 s to reduce preemption spam
        self._timer = self.create_timer(0.5, self._loop)

        self.get_logger().info('Vision-Aware Lane Follower Initialized.')

    # ------------------------------------------------------------------
    # Point-cloud callbacks
    # ------------------------------------------------------------------
    def _left_cb(self, msg):
        self._left_pts = self._unpack_cloud(msg)

    def _right_cb(self, msg):
        self._right_pts = self._unpack_cloud(msg)

    @staticmethod
    def _unpack_cloud(msg) -> np.ndarray:
        pts = list(point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
        if pts:
            return np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
        return np.empty((0, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _loop(self):
        if self._startup_sec > 0:
            elapsed = (self.get_clock().now() - self._start_wall).nanoseconds / 1e9
            if elapsed < self._startup_sec:
                return

        if not self._started:
            self._started = True
            self.get_logger().info('Starting Dynamic Vision Tracking!')

        # Genuine abort cooldown (not triggered by preemption)
        if self._last_abort_at is not None:
            since = (self.get_clock().now() - self._last_abort_at).nanoseconds / 1e9
            if since < self._abort_cooldown:
                return

        tf = self._get_robot_tf()
        if tf is None:
            return

        # Timeout on stalled goals
        if self._goal_active and self._goal_sent_at is not None:
            age = (self.get_clock().now() - self._goal_sent_at).nanoseconds / 1e9
            if age > self._timeout_sec:
                self.get_logger().warn('Goal timed out — cancelling.')
                self._cancel_goal_intentionally()

        if not self._goal_active or self._close_to_goal(tf):
            # No active goal or robot reached current one — always send new goal
            goal = self._compute_vision_goal(tf)
            if goal:
                self._send_goal(goal)
        else:
            # Goal is active and robot is still far — only update if the
            # lane centre has shifted significantly
            goal = self._compute_vision_goal(tf)
            if goal and self._goal_shifted_enough(goal):
                self._send_goal(goal)

    # ------------------------------------------------------------------
    # Goal computation — multi-slice centroid (same as v2)
    # ------------------------------------------------------------------
    def _compute_vision_goal(self, tf) -> PoseStamped:
        slices   = np.arange(self._step, self._lookahead + self._step * 0.5, self._step)
        best_z   = None
        best_cx  = None
        on_curve = False

        for z_target in slices:
            left_x  = self._wall_x_at_depth(self._left_pts,  z_target)
            right_x = self._wall_x_at_depth(self._right_pts, z_target)

            if left_x is not None and right_x is not None:
                cx = (left_x + right_x) / 2.0
                if best_z is None and abs(cx) > self._curve_thresh:
                    on_curve = True
                best_cx = cx
                best_z  = z_target
            elif (left_x is not None or right_x is not None) and best_z is None:
                best_cx = (left_x  + self._lane_width / 2.0) if left_x is not None \
                     else (right_x - self._lane_width / 2.0)
                best_z  = z_target

        # On curve: shorten lookahead to the nearer safe depth
        if on_curve and best_z is not None:
            best_z = min(best_z, self._curve_lookahead)
            lx = self._wall_x_at_depth(self._left_pts,  best_z)
            rx = self._wall_x_at_depth(self._right_pts, best_z)
            if lx is not None and rx is not None:
                best_cx = (lx + rx) / 2.0
            elif lx is not None:
                best_cx = lx + self._lane_width / 2.0
            elif rx is not None:
                best_cx = rx - self._lane_width / 2.0

        if best_z is None or best_cx is None:
            self.get_logger().warn('No lane points — blind fallback.',
                                   throttle_duration_sec=3.0)
            return self._compute_blind_goal(tf)

        best_cx = float(np.clip(best_cx, -self._lane_width * 0.6, self._lane_width * 0.6))

        pt_cam = PointStamped()
        pt_cam.header.frame_id = 'camera_link_optical'
        pt_cam.header.stamp    = rclpy.time.Time().to_msg()
        pt_cam.point.x = float(best_cx)
        pt_cam.point.y = 0.0
        pt_cam.point.z = float(best_z)

        try:
            pt_map = self._tf_buf.transform(pt_cam, 'map', timeout=Duration(seconds=0.5))
        except TransformException as exc:
            self.get_logger().warn(f'TF failed: {exc} — blind fallback.',
                                   throttle_duration_sec=3.0)
            return self._compute_blind_goal(tf)

        robot_x = tf.transform.translation.x
        robot_y = tf.transform.translation.y
        yaw = math.atan2(pt_map.point.y - robot_y, pt_map.point.x - robot_x)
        q   = quaternion_from_euler(0, 0, yaw)

        pose = PoseStamped()
        pose.header.frame_id       = 'map'
        pose.header.stamp          = self.get_clock().now().to_msg()
        pose.pose.position.x       = pt_map.point.x
        pose.pose.position.y       = pt_map.point.y
        pose.pose.position.z       = 0.0
        pose.pose.orientation.w    = q[0]
        pose.pose.orientation.x    = q[1]
        pose.pose.orientation.y    = q[2]
        pose.pose.orientation.z    = q[3]
        return pose

    # ------------------------------------------------------------------
    def _wall_x_at_depth(self, pts: np.ndarray, z_target: float):
        if pts.size == 0:
            return None
        mask  = (pts[:, 2] > z_target - self._z_tol) & \
                (pts[:, 2] < z_target + self._z_tol)
        valid = pts[mask]
        return float(np.median(valid[:, 0])) if valid.size > 0 else None

    def _compute_blind_goal(self, tf) -> PoseStamped:
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        pose = PoseStamped()
        pose.header.frame_id   = 'map'
        pose.header.stamp      = self.get_clock().now().to_msg()
        pose.pose.position.x   = x + self._curve_lookahead * math.cos(yaw)
        pose.pose.position.y   = y + self._curve_lookahead * math.sin(yaw)
        pose.pose.position.z   = 0.0
        pose.pose.orientation  = q
        return pose

    # ------------------------------------------------------------------
    # Goal helpers
    # ------------------------------------------------------------------
    def _close_to_goal(self, tf) -> bool:
        if self._current_goal is None:
            return True
        dx = tf.transform.translation.x - self._current_goal.pose.position.x
        dy = tf.transform.translation.y - self._current_goal.pose.position.y
        return math.hypot(dx, dy) < self._min_dist

    def _goal_shifted_enough(self, new_pose: PoseStamped) -> bool:
        """Only replace goal if lane centre has moved more than _update_dist."""
        if self._current_goal is None:
            return True
        dx = new_pose.pose.position.x - self._current_goal.pose.position.x
        dy = new_pose.pose.position.y - self._current_goal.pose.position.y
        return math.hypot(dx, dy) >= self._update_dist

    def _get_robot_tf(self):
        try:
            return self._tf_buf.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.2))
        except TransformException:
            return None

    def _send_goal(self, pose: PoseStamped):
        if not self._nav_client.wait_for_server(timeout_sec=2.0):
            return
        # Cancel the current goal INTENTIONALLY before sending the new one
        self._cancel_goal_intentionally()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(
            f'🎯 Vision Goal: ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})')

        future = self._nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_goal_accepted)
        self._goal_active  = True
        self._current_goal = pose
        self._goal_sent_at = self.get_clock().now()

    def _on_goal_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2.')
            self._goal_active = False
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        """
        Called when a goal finishes.

        STATUS_SUCCEEDED (4) → normal success
        STATUS_CANCELED  (5) → we cancelled it — no cooldown
        STATUS_ABORTED   (6) → Nav2 preempted it (new goal sent) OR genuine failure

        If _intentional_cancel is set we know we triggered the cancel — skip cooldown.
        Otherwise it is a genuine Nav2 failure (stuck, planner error) → cooldown.
        """
        if self._intentional_cancel:
            # We cancelled it on purpose to send a new goal — not a failure
            self._intentional_cancel = False
            self._goal_active = False
            self._goal_handle = None
            return

        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().debug('Goal succeeded.')
        elif status == GoalStatus.STATUS_CANCELED:
            pass   # normal cancel
        else:
            # Genuine Nav2 failure — back off and wait
            self.get_logger().warn(
                f'Goal genuinely failed (status={status}) — cooldown {self._abort_cooldown}s.')
            self._last_abort_at = self.get_clock().now()

        self._goal_active = False
        self._goal_handle = None

    def _cancel_goal_intentionally(self):
        """Cancel the current goal and flag it as intentional (not a failure)."""
        if self._goal_handle is not None:
            self._intentional_cancel = True
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None
        self._goal_active = False


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cancel_goal_intentionally()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()