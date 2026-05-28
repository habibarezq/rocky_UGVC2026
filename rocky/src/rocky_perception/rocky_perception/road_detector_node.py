#!/usr/bin/env python3
"""
Road Detector Node
==================
Detects lane markings and potholes from camera images and publishes
a merged PointCloud2 on /road_detector/pointcloud.

"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header, String

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs_py import point_cloud2

from rocky_perception.cv_code.road_features_detector import RoadFeatureDetector   # noqa: F401
from rocky_perception.cv_code.pipeline import RoadFeatureBEVPipeline


class RoadDetectorNode(Node):
    """ROS2 node for detecting road features from camera images."""

    def __init__(self):
        super().__init__('road_detector')

        self.bridge = CvBridge()

        self._declare_parameters()
        self._load_parameters()

        if not self._validate_parameters():
            self.get_logger().error(
                'Parameter validation failed — node may not function correctly')

        self.pipeline   = None
        self.image_size = (640, 480)   # updated from first frame
        self.latest_image = None

        self.processing_times = []
        self.frame_count      = 0

        self._setup_comms()
        self.add_on_set_parameters_callback(self._parameters_callback)

        self.get_logger().info('Road Detector Node initialised')
        self.get_logger().info(
            f'Camera: height={self.camera_height}m  '
            f'pitch={self.pitch_deg}°  '
            f'fx={self.K[0,0]:.1f}  fy={self.K[1,1]:.1f}')
        self.get_logger().info(
            f'Debug images: {self.publish_debug_images}')

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    def _declare_parameters(self):
        # Topics
        self.declare_parameter('camera_topic',              '/camera/image_raw')
        self.declare_parameter('camera_info_topic',         '/camera/camera_info')
        self.declare_parameter('output_pointcloud_topic',   '/road_detector/pointcloud')
        self.declare_parameter('output_lane_mask_topic',    '/road_detector/debug/lane_mask')
        self.declare_parameter('output_bev_topic',          '/road_detector/debug/bev_image')
        self.declare_parameter('output_stats_topic',        '/road_detector/stats')

        # ── Extrinsic camera parameters ────────────────────────────────
        # camera_height : vertical distance from ground to camera lens
        # pitch_deg     : negative = tilted downward (road-facing)
        self.declare_parameter('camera_height', 0.6)     # metres  (was 1.43 — car scale)
        self.declare_parameter('pitch_deg',    -35)    # degrees (was -50 — too steep for small robot)
        self.declare_parameter('yaw_deg',       0.0)
        self.declare_parameter('roll_deg',      0.0)

        # ── Intrinsic camera parameters ────────────────────────────────
        self.declare_parameter('fx', 554.0)   # (was 554.0 — matched 1920×1080)
        self.declare_parameter('fy', 554.0)
        self.declare_parameter('cx', 320.0)   # half of 640  (was 960)
        self.declare_parameter('cy', 240.0)   # half of 480  (was 540)
        self.declare_parameter('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])

        # Detection
        self.declare_parameter('min_radius', 5)    # smaller robot → smaller pothole radii in px
        self.declare_parameter('max_radius', 80)

        # Debug / performance
        self.declare_parameter('publish_debug_images',     False)
        self.declare_parameter('publish_performance_stats', False)

        # Point cloud cap
        self.declare_parameter('max_points_per_cloud', 10000)

    def _load_parameters(self):
        self.camera_topic    = self.get_parameter('camera_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.pc_topic        = self.get_parameter('output_pointcloud_topic').value
        self.lane_mask_topic = self.get_parameter('output_lane_mask_topic').value
        self.bev_topic       = self.get_parameter('output_bev_topic').value
        self.stats_topic     = self.get_parameter('output_stats_topic').value

        self.camera_height = self.get_parameter('camera_height').value
        self.pitch_deg     = self.get_parameter('pitch_deg').value
        self.yaw_deg       = self.get_parameter('yaw_deg').value
        self.roll_deg      = self.get_parameter('roll_deg').value

        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        self.dist_coeffs = self.get_parameter('dist_coeffs').value
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float64)

        self.min_radius    = self.get_parameter('min_radius').value
        self.max_radius    = self.get_parameter('max_radius').value

        self.publish_debug_images     = self.get_parameter('publish_debug_images').value
        self.publish_performance_stats = self.get_parameter('publish_performance_stats').value
        self.max_points_per_cloud     = self.get_parameter('max_points_per_cloud').value

        self.consecutive_error_counter = 0

    def _validate_parameters(self):
        valid = True

        if self.camera_height <= 0:
            self.get_logger().error(
                f'camera_height must be positive! Got: {self.camera_height}')
            valid = False

        if self.pitch_deg > 0:
            self.get_logger().warn(
                f'Positive pitch ({self.pitch_deg}°) points camera upward — '
                'road detection will likely fail')

        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        if fx <= 0 or fy <= 0:
            self.get_logger().error('Focal length (fx, fy) must be positive!')
            valid = False

        if self.min_radius >= self.max_radius:
            self.get_logger().error(
                f'min_radius ({self.min_radius}) must be < max_radius ({self.max_radius})')
            valid = False

        return valid

    # ------------------------------------------------------------------
    # Communications
    # ------------------------------------------------------------------
    def _setup_comms(self):
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)

        self.sub = self.create_subscription(
            Image, self.camera_topic, self._image_callback, image_qos)

        self.pc_pub = self.create_publisher(
            PointCloud2, self.pc_topic, reliable_qos)

        if self.publish_debug_images:
            self.img_pub = self.create_publisher(
                Image, self.lane_mask_topic, reliable_qos)
            self.bev_pub = self.create_publisher(
                Image, self.bev_topic, reliable_qos)

        if self.publish_performance_stats:
            self.stats_pub = self.create_publisher(
                String, self.stats_topic, reliable_qos)

    # ------------------------------------------------------------------
    # Pipeline initialisation
    # ------------------------------------------------------------------
    def _initialize_pipeline(self, image_width, image_height):
        if self.pipeline is not None:
            return

        self.get_logger().info(
            f'Initialising pipeline — image size: {image_width}×{image_height}')

        self.pipeline = RoadFeatureBEVPipeline(
            K=self.K,
            camera_height=self.camera_height,
            pitch_deg=self.pitch_deg,
            yaw_deg=self.yaw_deg,
            roll_deg=self.roll_deg,
            dist_coeffs=np.array(self.dist_coeffs).reshape(-1, 1),
            image_size=(image_width, image_height),
            min_radius=self.min_radius,
            max_radius=self.max_radius,
        )
        self.image_size = (image_width, image_height)
        self.get_logger().info('Pipeline initialisation complete')

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------
    def _image_callback(self, msg):
        self._process_image(msg)

    def _process_image(self, msg):
        start_time = self.get_clock().now()

        try:
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except CvBridgeError as e:
                self.get_logger().error(f'CV Bridge error: {e}')
                return

            if frame is None or frame.size == 0:
                self.get_logger().warn('Empty frame received')
                return

            h, w = frame.shape[:2]

            if self.pipeline is None:
                self._initialize_pipeline(w, h)

            try:
                (output, bev_image, lane_mask,
                 lane_points, ground_circles, circle_clouds) = \
                    self.pipeline.process_frame(frame)
            except Exception as e:
                self.get_logger().error(f'Pipeline error: {e}')
                self.consecutive_error_counter += 1
                return

            self.consecutive_error_counter = 0

            # ── Publish merged lane + circle point cloud ───────────────
            # All detected road-feature points go out on a single topic.
            
            all_points = list(lane_points) if len(lane_points) > 0 else []
            for cloud in circle_clouds:
                if len(cloud) > 0:
                    all_points.append(cloud)

            if all_points:
                merged = np.vstack(all_points)
            else:
                merged = np.empty((0, 3), dtype=np.float32)

            if len(merged) > self.max_points_per_cloud:
                # Uniform subsample to stay under cap
                idx    = np.random.choice(len(merged),
                                          self.max_points_per_cloud,
                                          replace=False)
                merged = merged[idx]
            # --- AXIS SWAP FIX ---
            # OpenCV assumes: X = Right, Y = Forward
            # ROS expects: X = Forward, Y = Left
            if len(merged) > 0:
                ros_merged = np.copy(merged)
                ros_merged[:, 0] = merged[:, 1]   # ROS X (Forward) = OpenCV Y (Depth)
                ros_merged[:, 1] = -merged[:, 0]  # ROS Y (Left) = -OpenCV X (Right)
                
                # --- THE OFFSET FIX ---
                # Shift the points forward by the physical distance 
                # from the base_footprint to the camera lens
                ros_merged[:, 0] += 0.305 
            else:
                ros_merged = merged

            pc_msg = self._make_pointcloud2(ros_merged, msg.header.frame_id)
            self.pc_pub.publish(pc_msg)
            self.get_logger().debug(f'Published {len(ros_merged)} road points')

            # ── Debug images ───────────────────────────────────────────
            if self.publish_debug_images:
                self._publish_debug_image(self.img_pub, lane_mask,  msg.header, 'lane_mask')
                self._publish_debug_image(self.bev_pub, bev_image,  msg.header, 'bev_image')

            # ── Performance stats ──────────────────────────────────────
            if self.publish_performance_stats:
                end_time      = self.get_clock().now()
                processing_ms = (end_time - start_time).nanoseconds / 1e6
                self._update_performance_stats(processing_ms)

            self.frame_count += 1
            if self.frame_count % 100 == 0:
                self.get_logger().info(
                    f'Processed {self.frame_count} frames successfully')

        except Exception as e:
            self.get_logger().error(
                f'Unexpected error processing frame: {e}',
                throttle_duration_sec=5.0)

    def _publish_debug_image(self, publisher, image, header, name):
        if image is None:
            self.get_logger().warn(
                f'{name} is None — skipping debug publish')
            return
        try:
            img_msg        = self.bridge.cv2_to_imgmsg(image, 'passthrough')
            img_msg.header = header
            publisher.publish(img_msg)
        except CvBridgeError as e:
            self.get_logger().error(
                f'CV Bridge error for {name} '
                f'(shape={image.shape}, dtype={image.dtype}): {e}')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_pointcloud2(self, points: np.ndarray, frame_id: str) -> PointCloud2:
        header            = Header()
        header.stamp      = self.get_clock().now().to_msg()
        header.frame_id   = frame_id

        if len(points) == 0:
            return point_cloud2.create_cloud_xyz32(header, np.zeros((0, 3)))

        pts = points.astype(np.float32)
        mask = np.isfinite(pts).all(axis=1)
        pts  = pts[mask]

        if len(pts) == 0:
            return point_cloud2.create_cloud_xyz32(header, np.zeros((0, 3)))

        return point_cloud2.create_cloud_xyz32(header, pts)

    def _update_performance_stats(self, processing_ms: float):
        self.processing_times.append(processing_ms)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)

        avg = np.mean(self.processing_times)
        std = np.std(self.processing_times)
        fps = 1000.0 / avg if avg > 0 else 0.0

        stats_msg      = String()
        stats_msg.data = (f'Frame: {self.frame_count}, '
                          f'Processing: {processing_ms:.2f}ms, '
                          f'Avg: {avg:.2f}ms ± {std:.2f}ms, '
                          f'FPS: {fps:.1f}')
        self.stats_pub.publish(stats_msg)

        if self.frame_count % 100 == 0:
            self.get_logger().info(stats_msg.data)

    def get_statistics(self):
        if not self.processing_times:
            return None
        avg = np.mean(self.processing_times)
        return {
            'frame_count':      self.frame_count,
            'avg_processing_ms': avg,
            'std_processing_ms': np.std(self.processing_times),
            'fps':               1000.0 / avg if avg > 0 else 0.0,
            'error_counter':     self.consecutive_error_counter,
        }

    # ------------------------------------------------------------------
    # Dynamic parameter callback
    # ------------------------------------------------------------------
    def _parameters_callback(self, params):
        result = rclpy.node.SetParametersResult(successful=True)

        for p in params:
            if p.name == 'publish_debug_images':
                self.publish_debug_images = p.value
            elif p.name == 'publish_performance_stats':
                self.publish_performance_stats = p.value
            elif p.name == 'max_points_per_cloud':
                self.max_points_per_cloud = p.value
            elif p.name == 'camera_height' and self.pipeline is not None:
                self.camera_height = p.value
                self.pipeline.set_camera_height(p.value)
            elif p.name == 'pitch_deg' and self.pipeline is not None:
                self.pitch_deg = p.value
                self.pipeline.set_pitch_deg(p.value)
            elif p.name == 'yaw_deg' and self.pipeline is not None:
                self.yaw_deg = p.value
                self.pipeline.set_yaw_deg(p.value)
            elif p.name == 'roll_deg' and self.pipeline is not None:
                self.roll_deg = p.value
                self.pipeline.set_roll_deg(p.value)
            elif p.name == 'min_radius' and self.pipeline is not None:
                self.min_radius = p.value
                self.pipeline.set_min_radius(p.value)
            elif p.name == 'max_radius' and self.pipeline is not None:
                self.max_radius = p.value
                self.pipeline.set_max_radius(p.value)
            elif p.name == 'dist_coeffs' and self.pipeline is not None:
                self.dist_coeffs = p.value
                self.pipeline.set_dist_coeffs(p.value)

            self.get_logger().info(f'Dynamic update: {p.name} = {p.value}')

        return result


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = RoadDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutdown requested by user')
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {e}')
    finally:
        stats = node.get_statistics()
        if stats:
            node.get_logger().info(f'Final statistics: {stats}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()