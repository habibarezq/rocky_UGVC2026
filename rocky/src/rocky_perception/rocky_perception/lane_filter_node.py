#!/usr/bin/env python3
"""
Lane Filter Node
================
Subscribes to a raw camera image, runs the RoadFeatureDetector pipeline,
and publishes:
  - /lane_points          (sensor_msgs/PointCloud2)  — ground-plane lane pixels
  - /virtual_walls/left   (sensor_msgs/PointCloud2)  — left lane wall
  - /virtual_walls/right  (sensor_msgs/PointCloud2)  — right lane wall
  - /lane_circles         (sensor_msgs/PointCloud2)  — circle contour clouds
  - /lane_image           (sensor_msgs/Image)        — annotated debug image
  - /bev_image            (sensor_msgs/Image)        — bird's-eye-view image

Camera intrinsics and extrinsics are loaded from ROS parameters so they can
be overridden at launch without recompiling.

Left / right wall splitting:
  Lane points with X < -lane_half_width are published on /virtual_walls/left,
  points with X >  lane_half_width on /virtual_walls/right, where X is the
  lateral ground coordinate (positive = right of camera optical axis).
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from road_features_detector import RoadFeatureDetector


# ---------------------------------------------------------------------------
# Helper: numpy (N,3) → PointCloud2
# ---------------------------------------------------------------------------
def array_to_cloud(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Convert an (N, 3) float32/float64 array to a PointCloud2 message."""
    header = Header()
    header.frame_id = frame_id
    header.stamp    = stamp

    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]

    pts = points.astype(np.float32)
    cloud = pc2.create_cloud(header, fields, pts)
    return cloud


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class LaneFilterNode(Node):

    def __init__(self):
        super().__init__('lane_filter_node')

        # ------------------------------------------------------------------
        # Declare parameters (can be set in launch file or YAML config)
        # ------------------------------------------------------------------
        self._declare_params()

        # ------------------------------------------------------------------
        # Build detector from parameters
        # ------------------------------------------------------------------
        self._detector = self._build_detector()

        # ------------------------------------------------------------------
        # CV Bridge
        # ------------------------------------------------------------------
        self._bridge = CvBridge()

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self._pub_lane      = self.create_publisher(PointCloud2, '/lane_points',         5)
        self._pub_left      = self.create_publisher(PointCloud2, '/virtual_walls/left',  5)
        self._pub_right     = self.create_publisher(PointCloud2, '/virtual_walls/right', 5)
        self._pub_circles   = self.create_publisher(PointCloud2, '/lane_circles',        5)
        self._pub_img       = self.create_publisher(Image,       '/lane_image',          5)
        self._pub_bev       = self.create_publisher(Image,       '/bev_image',           5)

        # ------------------------------------------------------------------
        # Subscriber
        # ------------------------------------------------------------------
        image_topic = self.get_parameter('image_topic').value
        self._sub = self.create_subscription(
            Image, image_topic, self._image_callback, 5)

        self.get_logger().info(
            f'LaneFilterNode ready — subscribing to [{image_topic}]')

    # ----------------------------------------------------------------------
    # Parameter declarations
    # ----------------------------------------------------------------------
    def _declare_params(self):
        def dp(name, value, description=''):
            descriptor = ParameterDescriptor(description=description)
            self.declare_parameter(name, value, descriptor)

        # Topics
        dp('image_topic', '/camera/image_raw', 'Input image topic')
        dp('camera_frame', 'camera_link_optical', 'Camera optical frame id')

        # Intrinsics — flat list [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        dp('K', [793.79768697, 0.0, 290.78702859,
                 0.0, 813.96117996, 241.57106901,
                 0.0, 0.0, 1.0],
           'Camera intrinsic matrix (row-major, 9 values)')

        # Distortion — [k1, k2, p1, p2, k3]
        dp('dist_coeffs',
           [-4.97661814e-01, 8.05356640e+00, 9.44660547e-03,
            -2.64434172e-02, -4.33974203e+01],
           'Distortion coefficients [k1,k2,p1,p2,k3]')

        # Extrinsics
        dp('camera_height', 1.33,   'Camera height above ground (m)')
        dp('pitch_deg',    -45.0,   'Camera pitch angle (degrees, negative = down)')
        dp('yaw_deg',       -2.0,   'Camera yaw angle (degrees)')
        dp('roll_deg',      -7.0,   'Camera roll angle (degrees)')

        # Image size — used to initialize homography; will be overridden
        # on first frame if it does not match the incoming stream.
        dp('image_width',   640, 'Expected image width  (pixels)')
        dp('image_height',  480, 'Expected image height (pixels)')

        # Lane geometry
        dp('lane_half_width', 0.375,
           'Half-width of lane (m). Points |X| > this split L/R wall.')

        # Detection limits
        dp('min_circle_radius', 10,  'Min circle radius in pixels')
        dp('max_circle_radius', 200, 'Max circle radius in pixels')

    # ----------------------------------------------------------------------
    # Build detector
    # ----------------------------------------------------------------------
    def _build_detector(self) -> RoadFeatureDetector:
        K_flat = self.get_parameter('K').value
        K = np.array(K_flat, dtype=np.float64).reshape(3, 3)

        dist = np.array(self.get_parameter('dist_coeffs').value, dtype=np.float64)

        w = self.get_parameter('image_width').value
        h = self.get_parameter('image_height').value

        detector = RoadFeatureDetector(
            K=K,
            camera_height=self.get_parameter('camera_height').value,
            pitch_deg=self.get_parameter('pitch_deg').value,
            yaw_deg=self.get_parameter('yaw_deg').value,
            roll_deg=self.get_parameter('roll_deg').value,
            image_size=(w, h),
            dist_coeffs=dist,
            min_radius=self.get_parameter('min_circle_radius').value,
            max_radius=self.get_parameter('max_circle_radius').value,
        )

        self._image_size = (w, h)
        self.get_logger().info(f'Detector initialised — image size: {w}x{h}')
        return detector

    # ----------------------------------------------------------------------
    # Reinitialise detector if image dimensions differ from expected
    # ----------------------------------------------------------------------
    def _check_reinit(self, h: int, w: int):
        if (w, h) != self._image_size:
            self.get_logger().warn(
                f'Image size changed to {w}x{h} — reinitialising detector.')
            self.set_parameters([
                rclpy.parameter.Parameter('image_width',  rclpy.Parameter.Type.INTEGER, w),
                rclpy.parameter.Parameter('image_height', rclpy.Parameter.Type.INTEGER, h),
            ])
            self._detector = self._build_detector()

    # ----------------------------------------------------------------------
    # Image callback
    # ----------------------------------------------------------------------
    def _image_callback(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        h, w = frame.shape[:2]
        self._check_reinit(h, w)

        stamp      = msg.header.stamp
        cam_frame  = self.get_parameter('camera_frame').value
        lane_half  = self.get_parameter('lane_half_width').value

        # ------------------------------------------------------------------
        # Run detector
        # ------------------------------------------------------------------
        output, edges, lines, ground_circles, circle_clouds, _ = \
            self._detector.process(frame, draw_bev=False)

        bev_image = self._detector.bev.warp_to_bev(frame)

        # ------------------------------------------------------------------
        # Lane mask → ground points
        # ------------------------------------------------------------------
        lane_mask = self._lines_to_mask(lines, frame.shape)
        pixels    = self._mask_to_pixels(lane_mask)

        if len(pixels) > 0:
            xy         = self._detector.bev.pixels_to_ground(pixels)
            z          = np.zeros((len(xy), 1), dtype=np.float64)
            lane_pts   = np.hstack([xy, z]).astype(np.float32)
        else:
            lane_pts = np.zeros((0, 3), dtype=np.float32)

        # ------------------------------------------------------------------
        # Split into left / right virtual walls
        # ------------------------------------------------------------------
        if len(lane_pts) > 0:
            # X axis: negative = left of robot, positive = right
            left_mask  = lane_pts[:, 0] < -lane_half
            right_mask = lane_pts[:, 0] >  lane_half
            left_pts   = lane_pts[left_mask]
            right_pts  = lane_pts[right_mask]
        else:
            left_pts  = np.zeros((0, 3), dtype=np.float32)
            right_pts = np.zeros((0, 3), dtype=np.float32)

        # ------------------------------------------------------------------
        # Circle clouds — concatenate all into one cloud
        # ------------------------------------------------------------------
        if circle_clouds:
            circle_pts = np.vstack(circle_clouds).astype(np.float32)
        else:
            circle_pts = np.zeros((0, 3), dtype=np.float32)

        # ------------------------------------------------------------------
        # Publish point clouds
        # ------------------------------------------------------------------
        self._pub_lane.publish(   array_to_cloud(lane_pts,    cam_frame, stamp))
        self._pub_left.publish(   array_to_cloud(left_pts,    cam_frame, stamp))
        self._pub_right.publish(  array_to_cloud(right_pts,   cam_frame, stamp))
        self._pub_circles.publish(array_to_cloud(circle_pts,  cam_frame, stamp))

        # ------------------------------------------------------------------
        # Publish debug images
        # ------------------------------------------------------------------
        try:
            self._pub_img.publish(self._bridge.cv2_to_imgmsg(output,    'bgr8'))
            self._pub_bev.publish(self._bridge.cv2_to_imgmsg(bev_image, 'bgr8'))
        except Exception as e:
            self.get_logger().error(f'Image publish error: {e}')

        self.get_logger().debug(
            f'lane={len(lane_pts)} left={len(left_pts)} '
            f'right={len(right_pts)} circles={len(ground_circles)}')

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _lines_to_mask(lines, shape):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        if lines is None:
            return mask
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=2)
        return mask

    @staticmethod
    def _mask_to_pixels(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([xs, ys], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LaneFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()