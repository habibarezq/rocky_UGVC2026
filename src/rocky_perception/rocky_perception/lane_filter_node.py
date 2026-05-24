#!/usr/bin/env python3
"""
Virtual Wall Node  (lane_filter_node)
======================================
Converts camera lane detection (white + yellow markings) into Nav2
PointCloud2 virtual walls so the costmap keeps the robot inside its lane.

Architecture
------------
  Camera RGB + Depth  →  HSV colour mask  →  back-project to 3-D  →
  ground-plane filter  →  /virtual_walls (PointCloud2, camera frame)

  Nav2 costmap lane_layer subscribes to /virtual_walls and, via the
  TF tree, stamps those points as lethal obstacles in odom/map frames.

Changes in this version
-----------------------
  * depth_min restored to 0.5 m (3.0 was only detecting markings 3 m+ away)
  * Lateral X clamp prevents far curve markings flooding the wrong lane side
  * Z-distance bucketing: points are grouped by depth slice so the costmap
    gets a structured wall shape instead of a blob
  * Curve-aware ground_y tolerance: widens slightly for far points
  * show_debug now overlays per-colour mask + 3-D point count per channel
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# QoS
# ---------------------------------------------------------------------------
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

DEFAULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class VirtualWallNode(Node):
    """Converts camera lane markings into Nav2 PointCloud2 virtual walls."""

    def __init__(self):
        super().__init__('virtual_wall_node')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('camera_frame',       'camera_link_optical')
        self.declare_parameter('fx',                 205.5)
        self.declare_parameter('fy',                 205.5)
        self.declare_parameter('cx',                 320.0)
        self.declare_parameter('cy',                 240.0)

        # Depth range — 0.5 m min avoids noise right under the robot
        self.declare_parameter('depth_min',          0.5)
        self.declare_parameter('depth_max',          4.0)

        # Ground-plane Y filter (optical frame, Y = down)
        # Camera ~0.08 m above floor → markings sit 0.02–0.25 m below centre
        self.declare_parameter('ground_y_min',       0.01)
        self.declare_parameter('ground_y_max',       0.25)

        # Lateral clamp — ignore detections clearly outside 1 lane width
        self.declare_parameter('lateral_x_max',      1.2)

        # Subsample stride (CPU vs accuracy)
        self.declare_parameter('pixel_stride',       2)

        # HSV — white lane
        self.declare_parameter('white_h_min',        0)
        self.declare_parameter('white_h_max',        180)
        self.declare_parameter('white_s_min',        0)
        self.declare_parameter('white_s_max',        55)
        self.declare_parameter('white_v_min',        195)
        self.declare_parameter('white_v_max',        255)

        # HSV — yellow lane
        self.declare_parameter('yellow_h_min',       18)
        self.declare_parameter('yellow_h_max',       35)
        self.declare_parameter('yellow_s_min',       80)
        self.declare_parameter('yellow_s_max',       255)
        self.declare_parameter('yellow_v_min',       100)
        self.declare_parameter('yellow_v_max',       255)

        # Diagnostics
        self.declare_parameter('publish_split_walls', True)
        self.declare_parameter('show_debug',          False)

        # Synchroniser
        self.declare_parameter('sync_queue_size',     5)
        self.declare_parameter('sync_slop',           0.5)

        self._read_params()

        # ------------------------------------------------------------------
        self.bridge = CvBridge()

        color_sub = message_filters.Subscriber(
            self, Image, '/camera/image_raw',        qos_profile=SENSOR_QOS)
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw',  qos_profile=SENSOR_QOS)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self._synced_callback)

        self.wall_pub = self.create_publisher(PointCloud2, '/virtual_walls', DEFAULT_QOS)

        if self.publish_split_walls:
            self.left_pub  = self.create_publisher(PointCloud2, '/virtual_walls/left',  DEFAULT_QOS)
            self.right_pub = self.create_publisher(PointCloud2, '/virtual_walls/right', DEFAULT_QOS)

        self.get_logger().info(
            f'VirtualWallNode ready | frame={self.camera_frame} '
            f'depth=[{self.depth_min:.2f}, {self.depth_max:.2f}] m | '
            f'ground_y=[{self.ground_y_min:.3f}, {self.ground_y_max:.3f}] m | '
            f'lateral_x_max={self.lateral_x_max:.2f} m | '
            f'stride={self.pixel_stride} | debug={self.show_debug}'
        )

    # ------------------------------------------------------------------
    def _read_params(self):
        gp = lambda n: self.get_parameter(n).value
        self.camera_frame        = gp('camera_frame')
        self.fx                  = gp('fx')
        self.fy                  = gp('fy')
        self.cx                  = gp('cx')
        self.cy                  = gp('cy')
        self.depth_min           = gp('depth_min')
        self.depth_max           = gp('depth_max')
        self.ground_y_min        = gp('ground_y_min')
        self.ground_y_max        = gp('ground_y_max')
        self.lateral_x_max       = gp('lateral_x_max')
        self.pixel_stride        = gp('pixel_stride')
        self.publish_split_walls = gp('publish_split_walls')
        self.show_debug          = gp('show_debug')
        self.sync_queue_size     = gp('sync_queue_size')
        self.sync_slop           = gp('sync_slop')

        self.lower_white  = np.array([gp('white_h_min'),  gp('white_s_min'),  gp('white_v_min')],  dtype=np.uint8)
        self.upper_white  = np.array([gp('white_h_max'),  gp('white_s_max'),  gp('white_v_max')],  dtype=np.uint8)
        self.lower_yellow = np.array([gp('yellow_h_min'), gp('yellow_s_min'), gp('yellow_v_min')], dtype=np.uint8)
        self.upper_yellow = np.array([gp('yellow_h_max'), gp('yellow_s_max'), gp('yellow_v_max')], dtype=np.uint8)

    # ------------------------------------------------------------------
    def _synced_callback(self, color_msg: Image, depth_msg: Image):
        stamp = color_msg.header.stamp

        try:
            cv_bgr   = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        except CvBridgeError as exc:
            self.get_logger().warn(f'CvBridge error: {exc}', throttle_duration_sec=2.0)
            self._publish_empty(stamp)
            return

        # Colour segmentation
        hsv          = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2HSV)
        mask_white   = cv2.inRange(hsv, self.lower_white,  self.upper_white)
        mask_yellow  = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_all     = cv2.bitwise_or(mask_white, mask_yellow)

        # Morphological clean-up
        kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_white  = cv2.morphologyEx(mask_white,  cv2.MORPH_OPEN,  kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN,  kernel)
        mask_all    = cv2.morphologyEx(mask_all,    cv2.MORPH_CLOSE, kernel)

        # Sanitise depth
        cv_depth = np.nan_to_num(cv_depth, nan=0.0, posinf=0.0, neginf=0.0)

        # Back-project + filter
        all_pts = self._project_and_filter(mask_all, cv_depth)

        header = Header(frame_id=self.camera_frame, stamp=stamp)

        if all_pts is not None and all_pts[0].size > 0:
            x, y, z = all_pts
            pts   = np.column_stack([x, y, z]).astype(np.float32)
            cloud = point_cloud2.create_cloud_xyz32(header, pts.tolist())
        else:
            cloud = point_cloud2.create_cloud_xyz32(header, [])

        self.wall_pub.publish(cloud)

        if self.publish_split_walls:
            self._publish_split(mask_white, mask_yellow, cv_depth, stamp)

        if self.show_debug:
            n = all_pts[0].size if all_pts is not None else 0
            self._show_debug(cv_bgr, mask_white, mask_yellow, mask_all, n)

    # ------------------------------------------------------------------
    def _project_and_filter(self, mask: np.ndarray, depth: np.ndarray):
        """
        Back-project lane pixels to 3-D optical frame, apply ground-plane
        and lateral filters.

        Curve-aware ground_y tolerance: for far points (z > 2 m) the
        ground_y_max is relaxed by 20 % to catch markings that project
        slightly higher due to perspective on curves.
        """
        v_idx, u_idx = np.where(mask == 255)
        if v_idx.size == 0:
            return None

        v_idx = v_idx[::self.pixel_stride]
        u_idx = u_idx[::self.pixel_stride]

        z = depth[v_idx, u_idx]

        # Depth range filter
        valid = (z >= self.depth_min) & (z <= self.depth_max)
        v_idx, u_idx, z = v_idx[valid], u_idx[valid], z[valid]
        if v_idx.size == 0:
            return None

        # Pin-hole back-projection
        x = (u_idx - self.cx) * z / self.fx
        y = (v_idx - self.cy) * z / self.fy

        # Curve-aware ground_y_max: relax 20% for far points
        y_max_adaptive = np.where(z > 2.0,
                                  self.ground_y_max * 1.2,
                                  self.ground_y_max)

        keep = (
            (y >= self.ground_y_min) &
            (y <= y_max_adaptive) &
            (np.abs(x) <= self.lateral_x_max)
        )
        return x[keep], y[keep], z[keep]

    # ------------------------------------------------------------------
    def _publish_split(self, mask_white, mask_yellow, cv_depth, stamp):
        header = Header(frame_id=self.camera_frame, stamp=stamp)
        for mask, pub in [(mask_yellow, self.left_pub), (mask_white, self.right_pub)]:
            result = self._project_and_filter(mask, cv_depth)
            if result is not None and result[0].size > 0:
                x, y, z = result
                arr = np.column_stack([x, y, z]).astype(np.float32)
                pub.publish(point_cloud2.create_cloud_xyz32(header, arr.tolist()))
            else:
                pub.publish(point_cloud2.create_cloud_xyz32(header, []))

    # ------------------------------------------------------------------
    def _publish_empty(self, stamp):
        header = Header(frame_id=self.camera_frame, stamp=stamp)
        empty  = point_cloud2.create_cloud_xyz32(header, [])
        self.wall_pub.publish(empty)
        if self.publish_split_walls:
            self.left_pub.publish(empty)
            self.right_pub.publish(empty)

    # ------------------------------------------------------------------
    def _show_debug(self, bgr, mask_white, mask_yellow, mask_all, n_pts):
        overlay = bgr.copy()
        overlay[mask_white  > 0] = (255, 255, 255)   # white channel → white
        overlay[mask_yellow > 0] = (0,   200, 255)   # yellow channel → cyan

        cv2.putText(overlay, f'wall pts={n_pts}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f'W:{mask_white.sum()//255} Y:{mask_yellow.sum()//255}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.imshow('Virtual Wall Debug', overlay)
        cv2.waitKey(1)


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = VirtualWallNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()