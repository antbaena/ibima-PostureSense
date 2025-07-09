import cv2
from rclpy.node import Node


class ArucoParams():
    def __init__(self, node:Node, aruco_params):
        if hasattr(cv2.aruco, aruco_params.aruco_dict):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_params.aruco_dict))
        else:
            node.get_logger().error(f"cv2.aruco doesn't have a dictionary with the name '{aruco_params.aruco_dict}'")
        self.marker_length = aruco_params.marker_length

