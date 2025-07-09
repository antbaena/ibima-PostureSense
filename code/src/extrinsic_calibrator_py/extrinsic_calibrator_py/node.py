#!/usr/bin/env python3
import rclpy
from .calibrator import ExtrinsicCalibrator
def main():
    rclpy.init()
    try:
        node = ExtrinsicCalibrator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down on user request.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
