# optimizer_node.py

import rclpy
from rclpy.node import Node

import numpy as np
from collections import defaultdict
from extrinsic_calibrator_interfaces.msg import MarkerObservation

import gtsam
from gtsam import Pose3, Rot3, Point3, BetweenFactorPose3
from gtsam.symbol_shorthand import X  # shorthand for variable naming


class OptimizerNode(Node):
    def __init__(self):
        super().__init__('optimizer_node')

        # Parameters
        self.declare_parameter('min_confidence', 0.3)
        self.min_confidence = self.get_parameter('min_confidence').value

        # Subscriptions
        self.subscription = self.create_subscription(
            MarkerObservation,
            '/marker_observations',
            self.observation_callback,
            100
        )

        # Internal storage
        self.observations = defaultdict(list)  # marker_id -> list[MarkerObservation]

        # Optimization timer
        self.timer = self.create_timer(5.0, self.optimize_graph)

        self.get_logger().info("🧠 OptimizerNode initialized and listening.")

    def observation_callback(self, msg: MarkerObservation):
        if msg.confidence < self.min_confidence:
            return

        self.observations[msg.marker_id].append(msg)

    def optimize_graph(self):
        if len(self.observations) < 2:
            self.get_logger().warn("⏳ Not enough markers to optimize.")
            return

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        marker_ids = sorted(self.observations.keys())
        fixed_id = marker_ids[0]

        # Fix the first marker at the origin
        initial.insert(X(fixed_id), Pose3(Rot3(), Point3()))
        noise_prior = gtsam.noiseModel.Diagonal.Variances(np.ones(6) * 1e-6)
        graph.add(gtsam.PriorFactorPose3(X(fixed_id), Pose3(), noise_prior))

        # Build connections between marker pairs seen from the same camera
        camera_ids = self._get_all_camera_ids()

        for cam_id in camera_ids:
            obs_by_cam = self._get_observations_by_camera(cam_id)
            marker_pairs = self._get_marker_pairs(obs_by_cam)

            for m1, m2 in marker_pairs:
                pose1 = self._average_transform(obs_by_cam[m1])
                pose2 = self._average_transform(obs_by_cam[m2])

                T_m1_m2 = pose1.between(pose2)

                avg_conf = (obs_by_cam[m1][0].confidence + obs_by_cam[m2][0].confidence) / 2
                sigma = 1.0 / max(avg_conf, 1e-3)
                noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * sigma)

                graph.add(BetweenFactorPose3(X(m1), X(m2), T_m1_m2, noise))

                if not initial.exists(X(m1)):
                    initial.insert(X(m1), pose1)
                if not initial.exists(X(m2)):
                    initial.insert(X(m2), pose2)

        # Run optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()

        self.get_logger().info("\n✅ Optimization complete. Estimated marker poses:")
        for marker_id in marker_ids:
            if result.exists(X(marker_id)):
                pose = result.atPose3(X(marker_id))
                t = pose.translation()
                self.get_logger().info(f"  ✉ Marker {marker_id}: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")


    def _get_all_camera_ids(self):
        return list({obs.camera_id for obs_list in self.observations.values() for obs in obs_list})

    def _get_observations_by_camera(self, cam_id):
        obs_by_marker = defaultdict(list)
        for marker_id, obs_list in self.observations.items():
            filtered = [o for o in obs_list if o.camera_id == cam_id]
            if filtered:
                obs_by_marker[marker_id] = filtered
        return obs_by_marker

    def _get_marker_pairs(self, marker_dict):
        ids = sorted(marker_dict.keys())
        return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]

    def _average_transform(self, obs_list):
        # Mean position
        positions = np.array([[obs.transform.translation.x,
                               obs.transform.translation.y,
                               obs.transform.translation.z] for obs in obs_list])
        mean_pos = np.mean(positions, axis=0)

        # Take orientation from the highest confidence
        best_obs = max(obs_list, key=lambda o: o.confidence)
        q = best_obs.transform.rotation
        rot = Rot3.Quaternion(q.w, q.x, q.y, q.z)

        return Pose3(rot, Point3(*mean_pos))


def main(args=None):
    rclpy.init(args=args)
    node = OptimizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
