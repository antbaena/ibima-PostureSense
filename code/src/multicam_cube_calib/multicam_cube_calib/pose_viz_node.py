#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

import numpy as np
import cv2
from cv_bridge import CvBridge

# MediaPipe
import mediapipe as mp
mp_pose = mp.solutions.pose

def depth_at(depth_img, u, v, win=3):
    """Devuelve la mediana de un parche alrededor del píxel (u,v) en metros; NaN si no hay datos."""
    h, w = depth_img.shape[:2]
    u0, v0 = int(round(u)), int(round(v))
    a = max(0, v0 - win)
    b = min(h, v0 + win + 1)
    c = max(0, u0 - win)
    d = min(w, u0 + win + 1)
    patch = depth_img[a:b, c:d].astype(np.float32)
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return np.nan
    # Si viene en mm, pásalo a metros: intenta detectar rango
    med = np.median(patch)
    if med > 10.0:  # probablemente mm
        med = med / 1000.0
    return float(med)

def backproject(u, v, z, fx, fy, cx, cy):
    """u,v en píxeles; z en metros -> punto 3D en cámara óptica."""
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return x, y, z

class PoseVizNode(Node):
    def __init__(self):
        super().__init__('pose_viz_node')

        # Parámetros
        self.declare_parameter('cameras', ['cam00/camera_00', 'cam00/camera_01'])
        self.declare_parameter('depth_topic_tpl', '/{}/depth/image_raw/decompressed')
        self.declare_parameter('camera_info_topic_tpl', '/{}/depth/camera_info')
        # NUEVO: color registrado/alineado a depth
        self.declare_parameter('color_topic_tpl', '/{}/color/image_raw/decompressed')

        self.cameras = self.get_parameter('cameras').get_parameter_value().string_array_value
        self.depth_tpl = self.get_parameter('depth_topic_tpl').get_parameter_value().string_value
        self.info_tpl  = self.get_parameter('camera_info_topic_tpl').get_parameter_value().string_value
        self.color_tpl = self.get_parameter('color_topic_tpl').get_parameter_value().string_value

        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.bridge = CvBridge()

        # Estructuras por cámara
        self.per_cam = {}
        for cam in self.cameras:
            color_topic = self.color_tpl.format(cam)
            depth_topic = self.depth_tpl.format(cam)
            info_topic  = self.info_tpl.format(cam)

            # Publisher de markers (uno por cámara -> namespaces distintos)
            pub = self.create_publisher(MarkerArray, f'/{cam}/pose_markers', 10)

            # Suscriptores
            sub_color = self.create_subscription(Image, color_topic,
                                lambda msg, cam=cam: self.on_color(msg, cam),
                                qos_img)
            sub_depth = self.create_subscription(Image, depth_topic,
                                lambda msg, cam=cam: self.on_depth(msg, cam),
                                qos_img)
            sub_info  = self.create_subscription(CameraInfo, info_topic,
                                lambda msg, cam=cam: self.on_info(msg, cam),
                                10)

            # Un estimador Pose por cámara (evita reinit por frame)
            pose = mp_pose.Pose(static_image_mode=False,
                                model_complexity=1,
                                enable_segmentation=False,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

            self.per_cam[cam] = {
                'pub': pub,
                'sub_color': sub_color,
                'sub_depth': sub_depth,
                'sub_info': sub_info,
                'pose': pose,
                'depth': None,
                'K': None,            # (fx, fy, cx, cy)
                'frame_id': None,     # de CameraInfo
                'last_stamp': None,
            }

            self.get_logger().info(f'[{cam}] color={color_topic} depth={depth_topic} info={info_topic} pub=/{cam}/pose_markers')

    def on_depth(self, msg: Image, cam: str):
        try:
            # depth puede venir en 16UC1 (mm) o 32FC1 (m)
            if msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.per_cam[cam]['depth'] = depth
        except Exception as e:
            self.get_logger().warn(f'[{cam}] error depth: {e}')

    def on_info(self, msg: CameraInfo, cam: str):
        try:
            K = msg.k  # fx, 0, cx, 0, fy, cy, 0, 0, 1
            fx, fy, cx, cy = K[0], K[4], K[2], K[5]
            self.per_cam[cam]['K'] = (fx, fy, cx, cy)
            # Usamos el frame_id de la cámara de color (viene del camera_info)
            self.per_cam[cam]['frame_id'] = msg.header.frame_id or 'camera_color_optical_frame'
        except Exception as e:
            self.get_logger().warn(f'[{cam}] error camera_info: {e}')

    def on_color(self, msg: Image, cam: str):
        pc = self.per_cam[cam]
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'[{cam}] error color: {e}')
            return

        # Ejecuta MediaPipe Pose
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pc['pose'].process(rgb)
        if not results.pose_landmarks:
            # publica markers vacíos para limpiar
            self.publish_clear(pc, msg.header, cam)
            return

        h, w = img.shape[:2]
        # Recolecta puntos 3D
        pts3d = []
        valid_idx = []
        fx, fy, cx, cy = (pc['K'] if pc['K'] is not None else (None, None, None, None))
        depth_img = pc['depth']
        for i, lm in enumerate(results.pose_landmarks.landmark):
            u = lm.x * w
            v = lm.y * h
            if depth_img is not None and fx is not None:
                z = depth_at(depth_img, u, v, win=2)
                if not np.isfinite(z) or z <= 0:
                    # fallback
                    z = 1.0
                x, y, z = backproject(u, v, z, fx, fy, cx, cy)
            else:
                # sin intrínsecos/depth: normaliza a z=1.0
                z = 1.0
                # usa intrínsecos virtuales centrados en imagen (evita NaNs)
                cx_v, cy_v = w * 0.5, h * 0.5
                fx_v, fy_v = max(w, 1), max(h, 1)
                x, y, z = backproject(u, v, z, fx_v, fy_v, cx_v, cy_v)

            pts3d.append((x, y, z))
            valid_idx.append(i)

        # Construye MarkerArray
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = pc['frame_id'] or (msg.header.frame_id or 'camera_color_optical_frame')

        markers = MarkerArray()
        ns = f'{cam}_pose'

        # Joints (SPHERE_LIST)
        m_pts = Marker()
        m_pts.header = header
        m_pts.ns = ns
        m_pts.id = 0
        m_pts.type = Marker.SPHERE_LIST
        m_pts.action = Marker.ADD
        m_pts.scale.x = 0.03  # diámetro ~3cm
        m_pts.scale.y = 0.03
        m_pts.scale.z = 0.03
        # Cada camara un color distinto
        if cam.endswith('00'):
            m_pts.color.r = 1.0
            m_pts.color.g = 0.2
            m_pts.color.b = 0.2
        else:
            m_pts.color.r = 0.2
            m_pts.color.g = 1.0
            m_pts.color.b = 1.0
        m_pts.color.a = 0.9
        m_pts.lifetime = rclpy.time.Duration(seconds=20.2).to_msg()

        from geometry_msgs.msg import Point
        m_pts.points = [Point(x=p[0], y=p[1], z=p[2]) for p in pts3d]
        markers.markers.append(m_pts)

        # Bones (LINE_LIST) usando conexiones de MediaPipe
        m_lines = Marker()
        m_lines.header = header
        m_lines.ns = ns
        m_lines.id = 1
        m_lines.type = Marker.LINE_LIST
        m_lines.action = Marker.ADD
        m_lines.scale.x = 0.01  # grosor
        # Cada camara un color distinto
        if cam.endswith('00'):
            m_lines.color.r = 1.0
            m_lines.color.g = 0.2
            m_lines.color.b = 0.2
        else:
            m_lines.color.r = 0.2
            m_lines.color.g = 1.0
            m_lines.color.b = 1.0
        m_lines.color.a = 0.9
        m_lines.lifetime = rclpy.time.Duration(seconds=20.2).to_msg()

        # Crea pares de índices desde POSE_CONNECTIONS
        try:
            connections = list(mp_pose.POSE_CONNECTIONS)
        except Exception:
            connections = []

        for a, b in connections:
            if a < len(pts3d) and b < len(pts3d):
                pa = pts3d[a]
                pb = pts3d[b]
                m_lines.points.append(Point(x=pa[0], y=pa[1], z=pa[2]))
                m_lines.points.append(Point(x=pb[0], y=pb[1], z=pb[2]))

        markers.markers.append(m_lines)

        pc['pub'].publish(markers)

    def publish_clear(self, pc, header_like, cam):
        header = Header()
        header.stamp = header_like.stamp
        header.frame_id = pc.get('frame_id') or (getattr(header_like, 'frame_id', '') or 'camera_color_optical_frame')

        arr = MarkerArray()
        for mid in (0, 1):
            m = Marker()
            m.header = header
            m.ns = f'{cam}_pose'
            m.id = mid
            m.action = Marker.DELETE
            arr.markers.append(m)
        pc['pub'].publish(arr)

def main():
    rclpy.init()
    node = PoseVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cierra limpiamente los objetos de MediaPipe
        for cam, data in node.per_cam.items():
            pose = data.get('pose')
            if pose:
                pose.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
