# camera_pose_optimizer_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import TransformStamped
import tf2_ros
import json

import numpy as np
from collections import defaultdict
from extrinsic_calibrator_interfaces.msg import MarkerObservation

import gtsam
from gtsam import Pose3, Rot3, Point3, BetweenFactorPose3
from gtsam.symbol_shorthand import C  # C for camera nodes


class CameraPoseOptimizer(Node):
    def __init__(self):
        super().__init__('camera_pose_optimizer')

        self.declare_parameter('min_confidence', 0.3)
        self.min_confidence = self.get_parameter('min_confidence').value

        self.observations = defaultdict(lambda: defaultdict(list))
        self.camera_id_map = {}

        self.optimization_result = {}
        self.result_available = False
        self.optimization_enabled = False

        self.create_subscription(
            MarkerObservation,
            '/marker_observations',
            self.observation_callback,
            100
        )

        self.publisher = self.create_publisher(String, '/camera_pose_metrics', 10)

        self.create_service(Trigger, '/start_optimization', self.start_optimization_callback)
        self.create_service(Trigger, '/finalize_result', self.finalize_result_callback)


        self.create_timer(5.0, self.optimize_camera_graph)
        self.get_logger().info('🎯 Camera pose optimizer node initialized.')

    def _get_camera_index(self, camera_id: str, frame_id: str = None) -> int:
        if camera_id not in self.camera_id_map:
            self.camera_id_map[camera_id] = {
                'index': len(self.camera_id_map),
                'frame_id': frame_id if frame_id else ""
            }
        elif frame_id and not self.camera_id_map[camera_id]['frame_id']:
            self.camera_id_map[camera_id]['frame_id'] = frame_id
        return self.camera_id_map[camera_id]['index']
    
    def _get_camera_frame_id(self, camera_id: str) -> str:
        return self.camera_id_map.get(camera_id, {}).get('frame_id', "")

    def observation_callback(self, msg: MarkerObservation):
        if  not self.optimization_enabled:
            return

        # Registrar el índice y el frame_id del camera_id
        self._get_camera_index(msg.camera_id, msg.header.frame_id)

        obs = {
            'camera_id': msg.camera_id,
            'frame_id': msg.header.frame_id,
            'marker_id': msg.marker_id,
            'confidence': msg.confidence,
            't': np.array([msg.transform.translation.x,
                        msg.transform.translation.y,
                        msg.transform.translation.z]),
            'q': np.array([msg.transform.rotation.w,
                        msg.transform.rotation.x,
                        msg.transform.rotation.y,
                        msg.transform.rotation.z])
        }
        self.observations[msg.marker_id][msg.camera_id].append(obs)

    def start_optimization_callback(self, request, response):
        # Reiniciar estado si ya estaba activa
        if self.optimization_enabled:
            self.get_logger().warn("🔄 Reiniciando optimización existente.")
            self._reset_state()

        self.optimization_enabled = True
        self.get_logger().info("Optimization process started.")
        response.success = True
        response.message = "Optimization started successfully"
        return response

    def _reset_state(self):
        """Reinicia todas las variables necesarias para comenzar de cero"""
        self.observations.clear()
        self.camera_id_map.clear()
        self.optimization_result = {}
        self.result_available = False
        self.optimization_enabled = False


    def finalize_result_callback(self, request, response):
        if not self.result_available:
            response.success = False
            response.message = "No result available to publish"
            return response

        # Publicar resultado en topic como string JSON
        msg = String()
        msg.data = json.dumps(self.optimization_result, indent=2)
        self.publisher.publish(msg)

        self.get_logger().info("Optimization result published.")
        response.success = True
        response.message = json.dumps(self.optimization_result)

        # Reiniciar estado interno para permitir nueva calibración
        self._reset_state()
        self.get_logger().info("Node reset and ready for new optimization.")

        return response


    def optimize_camera_graph(self):
        if not self.optimization_enabled:
            return

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        used_edges = set()
        cameras_set = set()

        for marker_id, cam_obs_dict in self.observations.items():
            cam_ids = list(cam_obs_dict.keys())
            for i in range(len(cam_ids)):
                for j in range(i + 1, len(cam_ids)):
                    cam_a = cam_ids[i]
                    cam_b = cam_ids[j]
                    key = tuple(sorted((cam_a, cam_b)))
                    if key in used_edges:
                        continue

                    T_m_in_a = self._average_pose(cam_obs_dict[cam_a])
                    T_m_in_b = self._average_pose(cam_obs_dict[cam_b])

                    T_a_to_b = T_m_in_a.inverse().compose(T_m_in_b)
                    confidence = (cam_obs_dict[cam_a][0]['confidence'] + cam_obs_dict[cam_b][0]['confidence']) / 2
                    noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * (1.0 / max(confidence, 1e-3)))

                    cam_a_idx = self._get_camera_index(cam_a)
                    cam_b_idx = self._get_camera_index(cam_b)

                    graph.add(BetweenFactorPose3(C(cam_a_idx), C(cam_b_idx), T_a_to_b, noise))

                    if not initial.exists(C(cam_a_idx)):
                        initial.insert(C(cam_a_idx), Pose3())
                    if not initial.exists(C(cam_b_idx)):
                        initial.insert(C(cam_b_idx), T_a_to_b)

                    cameras_set.update([cam_a, cam_b])
                    used_edges.add(key)

        if len(initial.keys()) == 0:
            self.get_logger().warn("⚠️ No valid camera pairs found. Waiting for more data...")
            return

        origin_id = sorted(list(cameras_set))[0]
        origin_idx = self._get_camera_index(origin_id)
        prior_noise = gtsam.noiseModel.Diagonal.Variances(np.ones(6) * 1e-6)
        graph.add(gtsam.PriorFactorPose3(C(origin_idx), Pose3(), prior_noise))

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()

        error_before = graph.error(initial)
        error_after = graph.error(result)

        self.optimization_result = {}
        for cam_id in sorted(cameras_set):
            cam_idx = self._get_camera_index(cam_id)
            cam_frame_id = self._get_camera_frame_id(cam_id)
            self.get_logger().info(f"Camera {cam_id} (index {cam_idx}) - Frame ID: {cam_frame_id}")
            if result.exists(C(cam_idx)):
                pose = result.atPose3(C(cam_idx))
                t = pose.translation()
                q = pose.rotation().toQuaternion()

                self.optimization_result[cam_id] = {
                    'frame_id': cam_frame_id,
                    'translation': [t[0], t[1], t[2]],
                    'rotation': [q.w(), q.x(), q.y(), q.z()]
                }

        log_msg = {
            "status": "success",
            "error_before": error_before,
            "error_after": error_after,
            "cameras": self.optimization_result
        }
        self.publisher.publish(String(data=json.dumps(log_msg)))
        self.result_available = True
        self.get_logger().info("✅ Optimización completa y lista para publicar.")

    def _average_pose(self, obs_list):
        positions = np.array([obs['t'] for obs in obs_list])
        mean_t = np.mean(positions, axis=0)
        best = max(obs_list, key=lambda x: x['confidence'])
        q = best['q']
        rot = Rot3.Quaternion(q[0], q[1], q[2], q[3])
        return Pose3(rot, Point3(*mean_t))


def main(args=None):
    rclpy.init(args=args)
    node = CameraPoseOptimizer()
    rclpy.spin(node)
