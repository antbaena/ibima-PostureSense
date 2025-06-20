#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
import message_filters
import numpy as np
from camera_postprocess import movenet_utils as mnu
import cv2 as cv

class MoveNet3DNode(Node):
    def __init__(self):
        super().__init__('movenet_3d_node')

        # Parámetros configurables
        self.declare_parameter('camera_names', ['camera_01', 'camera_02', 'camera_03', 'camera_04'])
        self.declare_parameter('model_url', 'https://tfhub.dev/google/movenet/singlepose/lightning/4')
        self.declare_parameter('input_size', 192)
        self.declare_parameter('keypoint_threshold', 0.3)
        self.declare_parameter('min_keypoints', 5)
        self.declare_parameter('min_depth', 0.1)  # metros
        self.declare_parameter('max_depth', 5.0)  # metros
        self.declare_parameter('annotated_topic', '/movenet/annotated_grid')

        # Leer parámetros
        cam_names = self.get_parameter('camera_names').get_parameter_value().string_array_value
        self.model_url = self.get_parameter('model_url').get_parameter_value().string_value
        self.input_size = self.get_parameter('input_size').get_parameter_value().integer_value
        self.kpt_thresh = self.get_parameter('keypoint_threshold').get_parameter_value().double_value
        self.min_kpts = self.get_parameter('min_keypoints').get_parameter_value().integer_value
        self.min_depth = self.get_parameter('min_depth').get_parameter_value().double_value
        self.max_depth = self.get_parameter('max_depth').get_parameter_value().double_value
        self.annotated_topic = self.get_parameter('annotated_topic').get_parameter_value().string_value

        # Derivar topics
        self.image_topics = [f"/{name}/color/image_raw" for name in cam_names]
        self.depth_topics = [f"/{name}/depth/image_raw" for name in cam_names]
        self.info_topics  = [f"/{name}/color/camera_info" for name in cam_names]
        self.marker_topics = [f"/{name}/skeleton_markers" for name in cam_names]

        # Cargar modelo
        self.get_logger().info(f"Cargando modelo MoveNet de {self.model_url}...")
        self.model = mnu.load_model(self.model_url)

        # Bridge e intrínsecas
        self.bridge = CvBridge()
        self.Ks = {name: None for name in cam_names}
        self.info_subs = {}

        # Definir conexiones del esqueleto COCO
        self.skel_conns = [
            (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),
            (6,8),(8,10),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)
        ]

        # Publishers de MarkerArray
        self.marker_pubs = {
            name: self.create_publisher(MarkerArray, topic, 10)
            for name, topic in zip(cam_names, self.marker_topics)
        }
        # Publisher mosaico anotado
        self.mosaic_pub = self.create_publisher(Image, self.annotated_topic, 10)

        # Suscripciones sincronizadas
        img_subs = [message_filters.Subscriber(self, Image, t) for t in self.image_topics]
        depth_subs = [message_filters.Subscriber(self, Image, t) for t in self.depth_topics]
        ts = message_filters.ApproximateTimeSynchronizer(img_subs + depth_subs, queue_size=10, slop=3.5)
        ts.registerCallback(self.callback)

        # Suscripción a CameraInfo
        for name, topic in zip(cam_names, self.info_topics):
            sub = self.create_subscription(
                CameraInfo, topic,
                lambda msg, n=name: self.info_callback(n, msg),
                10)
            self.info_subs[name] = sub

        self.get_logger().info(f"Nodo MoveNet3D inicializado con cámaras: {cam_names}")

    def info_callback(self, name, msg: CameraInfo):
        if self.Ks[name] is None:
            self.Ks[name] = msg
            self.get_logger().info(f"Intrínsecas recibidas para {name}")
            sub = self.info_subs.pop(name, None)
            if sub:
                self.destroy_subscription(sub)

    def create_mosaic(self, imgs):
        top = cv.hconcat([imgs[0], imgs[1]])
        bot = cv.hconcat([imgs[2], imgs[3]])
        return cv.vconcat([top, bot])

    def callback(self, *msgs):
        self.get_logger().info('Recibidos mensajes de imágenes y profundidad.')
        n = len(self.image_topics)
        imgs = [self.bridge.imgmsg_to_cv2(m, 'bgr8') for m in msgs[:n]]
        depths = [self.bridge.imgmsg_to_cv2(m, desired_encoding='passthrough') for m in msgs[n:]]
        headers = [m.header for m in msgs[:n]]
        cam_names = list(self.Ks.keys())

        annotated = []
        for idx, name in enumerate(cam_names):
            img = imgs[idx]
            depth = depths[idx]
            Kinfo = self.Ks[name]
            if Kinfo is None:
                annotated.append(img)
                continue

            # Inferencia y filtrado 2D
            kps_list, scs_list = mnu.run_inference_on_image(img, self.input_size, self.model)
            valid = mnu.process_detections(kps_list, scs_list, self.kpt_thresh, self.min_kpts)
            if not valid:
                annotated.append(img)
                continue
            keypoints, scores = valid[0]

            # Convertir 2D→3D
            pts3d = {}
            for i, ((u,v), sc) in enumerate(zip(keypoints, scores)):
                if sc < self.kpt_thresh: continue
                z = mnu.get_depth_value(u,v,depth)
                if z < self.min_depth or z > self.max_depth: continue
                X,Y,Z = mnu.convert_2d_to_3d(u,v,z,Kinfo)
                pts3d[i] = (X,Y,Z)

            # Crear MarkerArray
            markers = MarkerArray()
            # Puntos
            m_pts = Marker()
            m_pts.header = headers[idx]
            m_pts.header.frame_id = "map"
            m_pts.ns = f"pts_{name}"
            m_pts.id = 0
            m_pts.type = Marker.SPHERE_LIST
            m_pts.scale.x = m_pts.scale.y = m_pts.scale.z = 0.05
            m_pts.color.r = 0.0
            m_pts.color.g = 1.0
            m_pts.color.a = 1.0
            for pt in pts3d.values(): m_pts.points.append(self._to_point(pt))
            markers.markers.append(m_pts)
            # Líneas
            m_ln = Marker()
            m_ln.header = headers[idx]
            m_ln.header.frame_id = "map"
            m_ln.ns = f"ln_{name}"
            m_ln.id = 1
            m_ln.type = Marker.LINE_LIST
            m_ln.scale.x = 0.02
            m_ln.color.b = 1.0
            m_ln.color.a = 1.0
            for i,j in self.skel_conns:
                if i in pts3d and j in pts3d:
                    m_ln.points.append(self._to_point(pts3d[i]))
                    m_ln.points.append(self._to_point(pts3d[j]))
            markers.markers.append(m_ln)
            # Publicar
            self.marker_pubs[name].publish(markers)

            # Anotar imagen
            ann = mnu.draw_skeleton(img, keypoints, scores, self.kpt_thresh)
            annotated.append(ann)

        # Mosaico y publicación
        grid = self.create_mosaic(annotated)
        msg_img = self.bridge.cv2_to_imgmsg(grid, 'bgr8')
        msg_img.header = headers[0]
        self.mosaic_pub.publish(msg_img)
        self.get_logger().info('Publicado skeleton MarkerArrays y mosaico.')

    def _to_point(self, tpl):
        from geometry_msgs.msg import Point
        return Point(x=tpl[0], y=tpl[1], z=tpl[2])


def main(args=None):
    rclpy.init(args=args)
    node = MoveNet3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
