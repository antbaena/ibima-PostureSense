#!/usr/bin/env python3
"""
Preview Republisher Node — Reduce bandwidth for remote camera monitoring.

Subscribes to full-resolution CompressedImage (MJPG) topics published locally
by the camera driver, applies throttle + downscale + quality reduction, and
republishes a lightweight preview CompressedImage suitable for transmission
over bandwidth-constrained networks (e.g., RPi 100 Mbit Ethernet).

Bandwidth estimation:
  Original:  MJPG 640×480 @ 15fps  ≈  3   MB/s per camera
  Preview:   JPEG 320×240 @ 5fps   ≈  0.3 MB/s per camera (10× reduction)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class PreviewRepublisher(Node):
    """Throttle + downscale + recompress camera images for low-bandwidth preview."""

    def __init__(self):
        super().__init__('preview_republisher')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('input_topic', '')
        self.declare_parameter('output_topic', '')
        self.declare_parameter('target_fps', 5.0)
        self.declare_parameter('scale_factor', 0.5)
        self.declare_parameter('jpeg_quality', 50)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.target_fps = self.get_parameter('target_fps').value
        self.scale_factor = self.get_parameter('scale_factor').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value

        # ── Validate ────────────────────────────────────────────────
        if not input_topic:
            self.get_logger().fatal('Parameter "input_topic" is required')
            raise SystemExit(1)
        if not output_topic:
            # Derive from input: .../compressed → .../preview/compressed
            output_topic = input_topic.replace(
                '/compressed', '/preview/compressed'
            )
            self.get_logger().info(
                f'output_topic not set, derived: {output_topic}'
            )

        if self.target_fps <= 0:
            self.get_logger().fatal('target_fps must be > 0')
            raise SystemExit(1)
        if not (0.0 < self.scale_factor <= 1.0):
            self.get_logger().fatal('scale_factor must be in (0, 1]')
            raise SystemExit(1)
        if not (1 <= self.jpeg_quality <= 100):
            self.get_logger().fatal('jpeg_quality must be in [1, 100]')
            raise SystemExit(1)

        self._min_interval = 1.0 / self.target_fps
        self._last_pub_time = 0.0

        # ── Bridge ──────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Subscriber — BEST_EFFORT to match camera driver ────────
        self.sub = self.create_subscription(
            CompressedImage,
            input_topic,
            self._on_image,
            qos_profile_sensor_data,
        )

        # ── Publisher — BEST_EFFORT, VOLATILE, KEEP_LAST(1)
        # Remote subscribers never accumulate stale frames.
        preview_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(
            CompressedImage, output_topic, preview_qos
        )

        # ── Stats ───────────────────────────────────────────────────
        self._frames_in = 0
        self._frames_out = 0
        self.create_timer(10.0, self._log_stats)

        self.get_logger().info(
            f'PreviewRepublisher: {input_topic} → {output_topic}  '
            f'(fps={self.target_fps}, scale={self.scale_factor}, '
            f'quality={self.jpeg_quality})'
        )

    # ────────────────────────────────────────────────────────────────
    def _on_image(self, msg: CompressedImage):
        self._frames_in += 1

        # ── Throttle ────────────────────────────────────────────────
        now = time.monotonic()
        if (now - self._last_pub_time) < self._min_interval:
            return
        self._last_pub_time = now

        try:
            # Decode MJPG → cv::Mat
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warn_once(
                    'imdecode returned None — skipping frame'
                )
                return

            # ── Downscale ───────────────────────────────────────────
            if self.scale_factor < 1.0:
                new_w = int(image.shape[1] * self.scale_factor)
                new_h = int(image.shape[0] * self.scale_factor)
                image = cv2.resize(
                    image, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

            # ── Re-encode JPEG ──────────────────────────────────────
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            success, encoded = cv2.imencode('.jpg', image, encode_params)
            if not success:
                self.get_logger().warn('JPEG encode failed')
                return

            # ── Build output message ────────────────────────────────
            out_msg = CompressedImage()
            out_msg.header = msg.header
            out_msg.format = 'jpeg'
            out_msg.data = encoded.tobytes()
            self.pub.publish(out_msg)
            self._frames_out += 1

        except Exception as e:
            self.get_logger().error(f'Preview republisher error: {e}')

    # ────────────────────────────────────────────────────────────────
    def _log_stats(self):
        if self._frames_in > 0:
            self.get_logger().info(
                f'Preview stats: in={self._frames_in}, '
                f'out={self._frames_out}, '
                f'ratio={self._frames_out / self._frames_in:.1%}'
            )
            self._frames_in = 0
            self._frames_out = 0


def main(args=None):
    rclpy.init(args=args)
    node = PreviewRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down PreviewRepublisher...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
