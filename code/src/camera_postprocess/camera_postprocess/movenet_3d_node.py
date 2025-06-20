#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseArray
import message_filters
import numpy as np
from camera_postprocess import movenet_utils as mnu
import cv2 as cv

class MoveNet3DNode(Node):
    def __init__(self):
        super().__init__('movenet_3d_node')

        # Parámetros configurables
        self.declare_parameter('camera_names', ['camera_01', 'camera_02', 'camera_03', 'camera_04'])
        self.declare_parameter('model_url', "https://tfhub.dev/google/movenet/multipose/lightning/1")
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

        # Derivar topics de cada cámara (color, depth, info, pose)
        self.image_topics = [f"/{name}/color/image_raw" for name in cam_names]
        self.depth_topics = [f"/{name}/depth/image_raw" for name in cam_names]
        self.info_topics  = [f"/{name}/color/camera_info" for name in cam_names]
        self.pose_topics  = [f"/{name}/pose3d" for name in cam_names]

        # Cargar modelo MoveNet
        self.get_logger().info(f"Cargando modelo MoveNet de {self.model_url}...")
        self.model = mnu.load_model(self.model_url)

        # Bridge e intrínsecas por cámara
        self.bridge = CvBridge()
        self.Ks = {name: None for name in cam_names}

        # Publishers para cada cámara y mosaico
        self.pose_pubs = {
            name: self.create_publisher(PoseArray, topic, 10)
            for name, topic in zip(cam_names, self.pose_topics)
        }
        self.mosaic_pub = self.create_publisher(Image, self.annotated_topic, 10)

        # Suscribir imágenes y depth sincronizadas
        img_subs = [message_filters.Subscriber(self, Image, t) for t in self.image_topics]
        depth_subs = [message_filters.Subscriber(self, Image, t) for t in self.depth_topics]
        ts = message_filters.ApproximateTimeSynchronizer(img_subs + depth_subs, queue_size=10, slop=0.05)
        ts.registerCallback(self.callback)

        # Suscripciones a CameraInfo para cada color
        for name, topic in zip(cam_names, self.info_topics):
            self.create_subscription(
                CameraInfo, topic,
                lambda msg, n=name: self.info_callback(n, msg),
                10)

        self.get_logger().info(f"Nodo MoveNet3D inicializado con cámaras: {cam_names}")

    def info_callback(self, name, msg: CameraInfo):
        self.Ks[name] = msg
        self.get_logger().info(f"Matriz intrínseca recibida para {name}.")

    def create_mosaic(self, imgs):
        # Crea mosaico 2x2 de mismo tamaño
        top = cv.hconcat([imgs[0], imgs[1]])
        bot = cv.hconcat([imgs[2], imgs[3]])
        return cv.vconcat([top, bot])

    def callback(self, *msgs):
        n = len(self.image_topics)
        # Primeras n son color, siguientes n depth
        imgs = [self.bridge.imgmsg_to_cv2(m, 'bgr8') for m in msgs[:n]]
        depths = [self.bridge.imgmsg_to_cv2(m, desired_encoding='passthrough') for m in msgs[n:]]
        headers = [m.header for m in msgs[:n]]
        cam_names = list(self.Ks.keys())

        annotated = []
        # Procesar cada cámara por separado
        for idx, name in enumerate(cam_names):
            img = imgs[idx]
            depth = depths[idx]
            Kinfo = self.Ks[name]
            if Kinfo is None:
                self.get_logger().warning(f"Sin intrínsecas para {name}, omitiendo cámara.")
                annotated.append(img)
                continue

            # Detección MoveNet
            kps_list, scs_list = mnu.run_inference_on_image(img, self.input_size, self.model)
            valid = mnu.process_detections(kps_list, scs_list, self.kpt_thresh, self.min_kpts)
            if not valid:
                annotated.append(img)
                continue
            keypoints, scores = valid[0]

            # Reconstrucción 3D sin promediar, filtro por rango
            pts3d = []
            for (u, v), score in zip(keypoints, scores):
                if score < self.kpt_thresh:
                    pts3d.append((np.nan, np.nan, np.nan))
                    continue
                z = mnu.get_depth_value(u, v, depth)
                if z < self.min_depth or z > self.max_depth:
                    pts3d.append((np.nan, np.nan, np.nan))
                    continue
                X, Y, Z = mnu.convert_2d_to_3d(u, v, z, Kinfo)
                pts3d.append((X, Y, Z))

            # Publicar PoseArray individual
            pa = PoseArray()
            pa.header = headers[idx]
            for X, Y, Z in pts3d:
                pt = Point(x=X, y=Y, z=Z)
                pa.poses.append(pt)
            self.pose_pubs[name].publish(pa)

            # Anotar esqueleto en la imagen
            ann = mnu.draw_skeleton(img, keypoints, scores, self.kpt_thresh)
            annotated.append(ann)

        # Publicar mosaico de las 4 imágenes anotadas
        grid = self.create_mosaic(annotated)
        msg_img = self.bridge.cv2_to_imgmsg(grid, 'bgr8')
        msg_img.header = headers[0]
        self.mosaic_pub.publish(msg_img)
        self.get_logger().info("Publicado mosaico anotado y poses 3D individuales.")


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
