#!/usr/bin/env python3
import rclpy
import numpy as np
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Header
from multicam_cube_calib_interfaces.msg import PairMeasurement, CamerasMarkersList
from multicam_cube_calib.se3 import tf_to_mat, mat_to_tf


class PairBuilder(Node):
    """Construye pares camera->camera usando las detecciones sincronizadas
    de marcadores publicadas como `CamerasMarkersList`. Requiere un
    `cube_config` YAML que contenga la pose de cada marcador en el marco del
    cubo (marker -> cube).
    """

    def __init__(self):
        super().__init__('pair_builder')

        # Parámetros
        self.declare_parameter('markers_topic', '/camera_markers_sync')
        self.declare_parameter('cube_config', '/home/mapir/ibima-PostureSense/code/src/multicam_cube_calib/config/cube.yaml')
        self.declare_parameter('publish_topic', '/calib/pairs')

        self.markers_topic = self.get_parameter('markers_topic').get_parameter_value().string_value
        self.cube_config_path = self.get_parameter('cube_config').get_parameter_value().string_value
        self.publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value

        self.qos = QoSProfile(depth=10)

        # Cargar configuración del cubo (poses de marcadores)
        self.marker_to_cube = {}  # id -> 4x4 numpy matrix
        if self.cube_config_path:
            try:
                with open(self.cube_config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                for mid, v in (cfg.get('markers', {}) or {}).items():
                    try:
                        mid_i = int(mid)
                        if 'T' in v:
                            T = np.array(v['T'], dtype=float)
                            if T.shape == (4, 4):
                                self.marker_to_cube[mid_i] = T
                        elif 't' in v and 'q' in v:
                            t = np.array(v['t'], dtype=float)
                            q = np.array(v['q'], dtype=float)
                            # convertir quaternion + translation a 4x4
                            T = np.eye(4)
                            # quaternion q = [x,y,z,w]
                            x, y, z, w = q
                            # matriz de rotación
                            R = np.array([
                                [1-2*(y*y+z*z), 2*(x*y- z*w), 2*(x*z+ y*w)],
                                [2*(x*y+ z*w), 1-2*(x*x+z*z), 2*(y*z- x*w)],
                                [2*(x*z- y*w), 2*(y*z+ x*w), 1-2*(x*x+y*y)]
                            ], dtype=float)
                            T[:3, :3] = R
                            T[:3, 3] = t
                            self.marker_to_cube[mid_i] = T
                    except Exception:
                        self.get_logger().warning(f"Formato inválido para marker {mid} en cube_config; ignorando")
                self.get_logger().info(f"Cargadas {len(self.marker_to_cube)} poses de marcadores desde {self.cube_config_path}")
            except Exception as e:
                self.get_logger().warning(f"No se pudo cargar cube_config '{self.cube_config_path}': {e}")
        else:
            self.get_logger().warning("No se proporcionó parámetro 'cube_config'; no se podrán enlazar marcadores con el cubo")

        # Publisher y suscripción
        self.pub = self.create_publisher(PairMeasurement, self.publish_topic, 10)
        self.sub = self.create_subscription(CamerasMarkersList, self.markers_topic, self.on_markers, self.qos)
        self.get_logger().info(f"Suscrito a {self.markers_topic}, publicando pares en {self.publish_topic}")

    def on_markers(self, msg: CamerasMarkersList):
        #SI al menos dos camaras tienen detecciones continuar
        if sum(1 for cam_msg in msg.cameras if len(cam_msg.marker_ids) > 0) < 2:
            self.get_logger().debug("Menos de dos cámaras con detecciones, ignorando")
            return
        # Construir mapa cam -> t_cam_to_cube usando el mejor marcador disponible
        cam_to_cube = {}
        cam_conf = {}

        for cam_msg in msg.cameras:
            cam = cam_msg.camera
            best_score = -1.0
            best_T = None
            # recorrer detecciones de la cámara
            for mid, tform, conf in zip(cam_msg.marker_ids, cam_msg.t_cam_marker, cam_msg.confidence):
                if mid not in self.marker_to_cube:
                    continue
                try:
                    t_cam_marker = tf_to_mat(tform)
                    T_marker_to_cube = self.marker_to_cube[mid]
                    t_cam_to_cube = t_cam_marker @ T_marker_to_cube
                except Exception as e:
                    self.get_logger().warning(f"Error calculando T para cam {cam} marker {mid}: {e}")
                    continue
                if conf > best_score:
                    best_score = conf
                    best_T = t_cam_to_cube

            if best_T is not None:
                cam_to_cube[cam] = best_T
                cam_conf[cam] = float(max(1e-6, best_score))
                self.get_logger().info(f"Cámara {cam}: mejor marcador {mid} con confianza {best_score:.4f}")
        # Solo generar parejas para cámaras que comparten al menos un id de marcador visible
        cams = list(cam_to_cube.keys())
        for i in range(len(cams)):
            for j in range(i+1, len(cams)):
                ci = cams[i]
                cj = cams[j]

                Ti = cam_to_cube[ci]
                Tj = cam_to_cube[cj]
                Tij = Ti @ np.linalg.inv(Tj)

                pair = PairMeasurement()
                pair.header = Header()
                pair.header.stamp = msg.header.stamp if hasattr(msg, 'header') else self.get_clock().now().to_msg()
                pair.cam_i = ci
                pair.cam_j = cj
                pair.t_i_to_j = mat_to_tf(Tij)
                # peso: media geométrica de confidencias de ambas cámaras
                w = float(np.sqrt(cam_conf.get(ci, 1e-6) * cam_conf.get(cj, 1e-6)))
                pair.weight = w
                self.pub.publish(pair)
                self.get_logger().debug(f"Publicado par {ci} -> {cj} con peso {w:.4f}")
                # Publicar para debug la distancia entre cámaras en cm
                self.get_logger().debug(f"Distancia entre cámaras {ci} y {cj}: {np.linalg.norm(Tij[:3, 3]) * 100:.2f} cm")


def main():
    rclpy.init()
    node = PairBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
