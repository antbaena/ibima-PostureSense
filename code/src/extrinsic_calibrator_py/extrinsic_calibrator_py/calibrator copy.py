#!/usr/bin/env python3
"""
Module for extrinsic calibration of multiple cameras using ArUco markers.
Refactored and modularized following best practices.
"""
import time
import logging
from typing import List, Dict, Tuple
from itertools import combinations, deque

import numpy as np
import pandas as pd
import tf2_ros
import tf_transformations as tft
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
import rclpy

from .camera import Camera

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExtrinsicCalibrator(Node):
    """
    Calibrates extrinsic transforms between multiple cameras using ArUco markers.
    """

    TIMER_PERIOD_S = 2.0

    def __init__(self) -> None:
        super().__init__("extrinsic_calibrator")
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Load and validate configuration
        params = self._load_parameters()
        self.camera_configs = self._build_camera_configs(params)
        self.array_of_cameras = self._init_cameras(self.camera_configs, params)

        self._log_camera_setup()

        # Periodic check for camera readiness
        self.create_timer(self.TIMER_PERIOD_S, self._check_ready_callback)

    def _load_parameters(self) -> Dict[str, any]:
        """Load ROS parameters and return as dict."""
        self.declare_parameter("aruco_dictionary", "DICT_6X6_250")
        self.declare_parameter("marker_length", 0.25)
        self.declare_parameter("camera_names", [])
        self.declare_parameter("image_topics", [])
        self.declare_parameter("info_topics", [])
        self.declare_parameter("frame_ids", [])

        aruco_dict = (
            self.get_parameter("aruco_dictionary").get_parameter_value().string_value
        )
        marker_length = (
            self.get_parameter("marker_length").get_parameter_value().double_value
        )
        names = (
            self.get_parameter("camera_names").get_parameter_value().string_array_value
        )
        imgs = (
            self.get_parameter("image_topics").get_parameter_value().string_array_value
        )
        infos = (
            self.get_parameter("info_topics").get_parameter_value().string_array_value
        )
        frames = (
            self.get_parameter("frame_ids").get_parameter_value().string_array_value
        )

        if not (len(names) == len(imgs) == len(infos) == len(frames) >= 1):
            self.get_logger().error("Camera parameters lists must be same length >=1.")
            self.destroy_node()
            raise RuntimeError("Invalid camera configuration parameters.")

        return {
            "aruco_dictionary": aruco_dict,
            "marker_length": marker_length,
            "camera_names": names,
            "image_topics": imgs,
            "info_topics": infos,
            "frame_ids": frames,
        }

    def _build_camera_configs(self, params: Dict[str, any]) -> List[Dict[str, str]]:
        """Construct list of camera config dicts."""
        configs = []
        for name, img, info, frame in zip(
            params["camera_names"],
            params["image_topics"],
            params["info_topics"],
            params["frame_ids"],
        ):
            configs.append(
                {
                    "name": name,
                    "image_topic": img,
                    "info_topic": info,
                    "frame_id": frame,
                }
            )
        return configs

    def _init_cameras(
        self,
        configs: List[Dict[str, str]],
        params: Dict[str, any],
    ) -> List[Camera]:
        """Instantiate Camera objects based on configurations."""
        cameras: List[Camera] = []
        for idx, cfg in enumerate(configs):
            cam = Camera(
                node=self,
                camera_name=cfg["name"],
                camera_id=idx,
                image_topic=cfg["image_topic"],
                camera_info_topic=cfg["info_topic"],
                marker_length=params["marker_length"],
                aruco_dict_name=params["aruco_dictionary"],
                camera_frame_id=cfg["frame_id"],
            )
            cameras.append(cam)
        return cameras

    def _log_camera_setup(self) -> None:
        """Log the camera setup details."""
        for cam in self.array_of_cameras:
            self.get_logger().info(
                f"Camera {cam.camera_id}: '{cam.camera_name}', "
                f"img_topic='{cam.image_topic}', info_topic='{cam.camera_info_topic}', "
                f"frame_id='{cam.camera_frame_id}'"
            )

    def _check_ready_callback(self) -> None:
        """Timer callback: checks if all cameras ready to calibrate."""
        if all(cam.are_all_transforms_precise() for cam in self.array_of_cameras):
            self.get_logger().info("All marker transforms gathered.")
            self._cleanup_subscriptions()
            self._run_calibration_routine()
        else:
            for cam in self.array_of_cameras:
                if cam.camera_matrix is None or cam.dist_coeffs is None:
                    self.get_logger().warn(
                        f"Camera '{cam.camera_name}' parameters missing"
                    )
            self.get_logger().warn("Waiting for all cameras to be ready.")

    def _cleanup_subscriptions(self) -> None:
        """Remove image and info subscriptions after data collection."""
        for cam in self.array_of_cameras:
            self.destroy_subscription(cam.image_sub)
            if cam.camera_info_sub:
                self.destroy_subscription(cam.camera_info_sub)
            cam._remove_unreliable_markers()

    def _run_calibration_routine(self) -> None:
        """Execute calibration steps sequentially."""
        steps = [
            (self._find_max_marker_id, "Find max marker ID"),
            (self._display_common_markers, "Display common markers"),
            (self._check_graph_connectivity, "Check graph connectivity"),
            (self._display_pairwise_transforms, "Display pairwise transforms"),
            (self._compute_pairwise_calibration, "Compute pairwise calibration"),
            (self._display_tf_chain, "Display TF chain"),
        ]

        for func, desc in steps:
            self.get_logger().info(desc)
            if not func():
                self.get_logger().error(f"Step failed: {desc}")
                return

        self.get_logger().info("Extrinsic calibration completed successfully.")
        self._hold_node()

    def _hold_node(self) -> None:
        """Keep node alive to maintain published TFs."""
        try:
            while rclpy.ok():
                time.sleep(1)
        except KeyboardInterrupt:
            self.get_logger().info("Shutting down.")

    def _find_max_marker_id(self) -> bool:
        """Determine the highest marker ID observed."""
        all_ids = [m.id for cam in self.array_of_cameras for m in cam.markers.values()]
        self.max_marker_id = max(all_ids, default=0)
        self.get_logger().info(f"Max marker ID: {self.max_marker_id}")
        return True

    def _compute_common_markers(self) -> pd.DataFrame:
        """Return DataFrame of common marker IDs between camera pairs."""
        rows = []
        for cam_a, cam_b in combinations(self.array_of_cameras, 2):
            common = set(cam_a.markers) & set(cam_b.markers)
            if common:
                rows.append(
                    {
                        "cam_a": cam_a.camera_name,
                        "cam_b": cam_b.camera_name,
                        "common_markers": sorted(common),
                    }
                )
        return pd.DataFrame(rows, columns=["cam_a", "cam_b", "common_markers"])

    def _display_common_markers(self) -> bool:
        """Log common markers table to console."""
        df = self._compute_common_markers()
        if df.empty:
            self.get_logger().info("No common markers found.")
            return False
        table = df.to_string(index=False)
        self.get_logger().info(f"\n{table}")
        self.common_df = df
        return True

    def _check_graph_connectivity(self) -> bool:
        """Ensure the camera graph is fully connected."""
        names = [cam.camera_name for cam in self.array_of_cameras]
        idx = {n: i for i, n in enumerate(names)}
        N = len(names)
        adj = {i: set() for i in range(N)}
        for _, row in self.common_df.iterrows():
            a, b = idx[row["cam_a"]], idx[row["cam_b"]]
            adj[a].add(b)
            adj[b].add(a)

        visited = set()
        queue = deque([0])
        visited.add(0)
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        if len(visited) == N:
            self.get_logger().info("All cameras are connected.")
            return True
        missing = [names[i] for i in range(N) if i not in visited]
        self.get_logger().error(f"Disconnected cameras: {missing}")
        return False

    def _display_pairwise_transforms(self) -> bool:
        """Compute and log raw pairwise marker transforms."""
        rows = []
        name_map = {cam.camera_name: cam for cam in self.array_of_cameras}
        for _, r in self.common_df.iterrows():
            cam_a = name_map[r["cam_a"]]
            cam_b = name_map[r["cam_b"]]
            for mid in r["common_markers"]:
                M_a = cam_a.markers[mid].tf
                M_b = cam_b.markers[mid].tf
                M_ab = M_a @ np.linalg.inv(M_b)
                t_ab = M_ab[:3, 3]
                q_ab = tft.quaternion_from_matrix(M_ab)
                rows.append(
                    {
                        "cam_a": cam_a.camera_name,
                        "cam_b": cam_b.camera_name,
                        "marker_id": mid,
                        **{f"t{i}": t_ab[i] for i in range(3)},
                        **{f"q{i}": q_ab[i] for i in range(4)},
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            self.get_logger().info("No pairwise transforms.")
            return False
        self.get_logger().info(f"\n{df.to_string(index=False)}")
        self.raw_pairwise_df = df
        return True

    def _compute_pairwise_calibration(self) -> bool:
        """Aggregate and validate pairwise transforms, prepare for chain build."""
        rows = []
        grouped = self.raw_pairwise_df.groupby(["cam_a", "cam_b"])
        for (a, b), group in grouped:
            t_arr = np.stack(group[["t0", "t1", "t2"]].values)
            q_arr = np.stack(group[["q0", "q1", "q2", "q3"]].values)
            mean_t = t_arr.mean(axis=0)
            q_mean = q_arr.sum(axis=0)
            mean_q = q_mean / np.linalg.norm(q_mean)
            max_dist = np.linalg.norm(t_arr - mean_t, axis=1).max()
            dots = np.abs(np.dot(q_arr, mean_q))
            max_ang = np.degrees((2 * np.arccos(np.clip(dots, -1, 1))).max())
            rows.append(
                {
                    "cam_a": a,
                    "cam_b": b,
                    "num_markers": len(group),
                    "mean_t": mean_t,
                    "mean_q": mean_q,
                    "max_dist": max_dist,
                    "max_ang": max_ang,
                }
            )
        self.pairwise_df = pd.DataFrame(rows)
        self.get_logger().info(f"Computed {len(self.pairwise_df)} pairwise transforms.")
        return True

    def _build_tf_chain(self) -> pd.DataFrame:
        """Construct and publish the static TF chain, return chain DataFrame."""
        # Build map of transforms
        tf_map: Dict[Tuple[str, str], np.ndarray] = {}
        for _, r in self.pairwise_df.iterrows():
            M = tft.quaternion_matrix(r["mean_q"])  # type: ignore
            M[:3, 3] = r["mean_t"]  # type: ignore
            tf_map[(r["cam_a"], r["cam_b"])] = M
            tf_map[(r["cam_b"], r["cam_a"])] = np.linalg.inv(M)

        # BFS tree
        root = self.array_of_cameras[0].camera_name
        visited = {root}
        queue = deque([root])
        parents: Dict[str, Tuple[str, np.ndarray]] = {}
        while queue:
            src = queue.popleft()
            for (a, b), M in tf_map.items():
                if a == src and b not in visited:
                    visited.add(b)
                    parents[b] = (a, M)
                    queue.append(b)

        # Publish and record chain
        rows = []
        for child, (parent, M) in parents.items():
            t = M[:3, 3]
            q = tft.quaternion_from_matrix(M)
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = parent
            tf_msg.child_frame_id = child
            (
                tf_msg.transform.translation.x,
                tf_msg.transform.translation.y,
                tf_msg.transform.translation.z,
            ) = t
            (
                tf_msg.transform.rotation.x,
                tf_msg.transform.rotation.y,
                tf_msg.transform.rotation.z,
                tf_msg.transform.rotation.w,
            ) = q
            self.tf_broadcaster.sendTransform(tf_msg)
            rows.append(
                {
                    "parent": parent,
                    "child": child,
                    "translation": t,
                    "rotation": q,
                }
            )
        return pd.DataFrame(rows)

    def _display_tf_chain(self) -> bool:
        """Log the published TF chain."""
        df = self._build_tf_chain()
        if df.empty:
            self.get_logger().info("No TF chain to display.")
            return False
        self.get_logger().info(f"\n{df.to_string(index=False)}")
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExtrinsicCalibrator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
