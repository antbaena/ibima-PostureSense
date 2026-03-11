#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
import message_filters
import numpy as np
import cv2 as cv
import time
from functools import partial
from camera_postprocess import movenet_utils as mnu

# QoS for subscribing to camera sensor data (must match publisher: BEST_EFFORT)
sensor_qos = qos_profile_sensor_data

# QoS for publishing processed results (RELIABLE for downstream consumers)
defensive_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST
)

class MoveNet3DNode(Node):
    def __init__(self):
        super().__init__('movenet_3d_node')
        # cb_group = ReentrantCallbackGroup()

        # ---------------------- Parámetros ----------------------
        self.declare_parameter('camera_names', [
            'cam00/camera_00', 'cam00/camera_01',
            'cam01/camera_02', 'cam02/camera_03'
        ])
        self.declare_parameter('model_url', 'https://tfhub.dev/google/movenet/singlepose/lightning/4')
        self.declare_parameter('input_size', 192)
        self.declare_parameter('keypoint_threshold', 0.3)
        self.declare_parameter('min_keypoints', 5)
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 5.0)
        self.declare_parameter('processing_frequency', 30.0)
        self.declare_parameter('publish_mosaic', False)  # ya no necesitamos mosaico

        # ------------------------ Obtener parámetros ------------------------
        cam_names = self.get_parameter('camera_names').value
        self.model_url = self.get_parameter('model_url').value
        self.input_size = self.get_parameter('input_size').value
        self.kpt_thresh = self.get_parameter('keypoint_threshold').value
        self.min_kpts = self.get_parameter('min_keypoints').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.proc_freq = self.get_parameter('processing_frequency').value

        self._proc_interval = 1.0 / self.proc_freq if self.proc_freq > 0 else 0.0
        self._last_proc_time = {name: 0.0 for name in cam_names}

        # ------------------------- Topics setup -------------------------
        self.bridge = CvBridge()
        # Publishers de imagen anotada
        self.annot_pubs = {
            name: self.create_publisher(
                Image,
                f"/{name}/annotated/image_raw",
                qos_profile=defensive_qos
            )
            for name in cam_names
        }
        # Publishers de marcadores esqueléticos
        self.marker_pubs = {
            name: self.create_publisher(
                MarkerArray,
                f"/{name}/skeleton_markers",
                qos_profile=defensive_qos
            )
            for name in cam_names
        }

        # ------------------------ Cargar modelo ------------------------
        self.get_logger().info(f"Cargando modelo MoveNet desde {self.model_url}...")
        self.model = mnu.load_model(self.model_url)

        # Intrínsecas por cámara
        self.Ks = {name: None for name in cam_names}

        # ---------------------- Suscripciones ----------------------
        for name in cam_names:
            # CameraInfo — use sensor_data QoS to match camera driver
            info_topic = f"/{name}/color/camera_info"
            self.create_subscription(
                CameraInfo,
                info_topic,
                partial(self.info_callback, name),
                qos_profile=sensor_qos,
            )

            # Suscriptores de imagen color y profundidad — BEST_EFFORT to match camera driver
            img_sub = message_filters.Subscriber(self, CompressedImage, f"/{name}/color/image_raw/compressed", qos_profile=sensor_qos)
            depth_sub = message_filters.Subscriber(self, CompressedImage, f"/{name}/depth/image_raw/compressedDepth", qos_profile=sensor_qos)

            # Sincronizador por par (RGB + profundidad)
            sync = message_filters.ApproximateTimeSynchronizer(
                [img_sub, depth_sub],
                queue_size=5,
                slop=0.1  # tolerancia pequeña
            )
            sync.registerCallback(partial(self.camera_callback, name))

        # Conexiones esqueléticas (COCO)
        self.skel_conns = [
            (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),
            (6,8),(8,10),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)
        ]

        self.get_logger().info(
            f"MoveNet3DNode inicializado con cámaras={cam_names}, proc_freq={self.proc_freq}Hz"
        )

    def info_callback(self, name, msg: CameraInfo):
        if self.Ks[name] is None:
            self.Ks[name] = msg
            self.get_logger().info(f"Intrínsecas recibidas para {name}")

    def camera_callback(self, name, img_msg: CompressedImage, depth_msg: CompressedImage):
        # Asegurarnos de tener intrínsecas
        if self.Ks[name] is None:
            self.get_logger().warn(f"Esperando intrínsecas de {name}")
            return

        # Throttle por cámara
        stamp = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
        if 0 < self._proc_interval and (stamp - self._last_proc_time[name]) < self._proc_interval:
            return
        self._last_proc_time[name] = stamp

        # Decodificar imágenes
        img = self.bridge.compressed_imgmsg_to_cv2(img_msg, 'bgr8')
        raw = depth_msg.data[12:]
        arr = np.frombuffer(raw, np.uint8)
        depth = cv.imdecode(arr, cv.IMREAD_UNCHANGED)
        if depth is None:
            depth = np.zeros((1,1), dtype=np.uint16)

        # Inferencia 2D
        kps_list, scs_list = mnu.run_inference_on_image(img, self.input_size, self.model)
        valid = mnu.process_detections(kps_list, scs_list, self.kpt_thresh, self.min_kpts)
        kps, scores = valid[0] if valid else (None, None)

        # Procesamiento 3D y publicación
        if kps is not None:
            # Construir puntos 3D
            pts3d = {}
            for i, ((u,v), sc) in enumerate(zip(kps, scores)):
                if sc < self.kpt_thresh:
                    continue
                z = mnu.get_depth_value(u, v, depth)
                if not (self.min_depth <= z <= self.max_depth):
                    continue
                pts3d[i] = mnu.convert_2d_to_3d(u, v, z, self.Ks[name])

            # Publicar marcadores
            markers = self._make_markers(pts3d, img_msg.header, name)
            self.marker_pubs[name].publish(markers)
            # Dibujar esqueleto en la imagen
            annotated = mnu.draw_skeleton(img.copy(), kps, scores, self.kpt_thresh)
        else:
            annotated = img

        # Publicar imagen anotada
        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = img_msg.header
        self.annot_pubs[name].publish(out_msg)

    def _make_markers(self, pts3d, header, name):
        markers = MarkerArray()
        from geometry_msgs.msg import Point

        # Puntos
        m_pts = Marker(
            header=header,
            ns=f"pts_{name}",
            id=0,
            type=Marker.SPHERE_LIST,
        )
        m_pts.scale.x = m_pts.scale.y = m_pts.scale.z = 0.05
        m_pts.color.r = 0.0; m_pts.color.g = 1.0; m_pts.color.a = 1.0
        for pt in pts3d.values():
            p = Point(x=pt[0], y=pt[1], z=pt[2])
            m_pts.points.append(p)
        markers.markers.append(m_pts)

        # Líneas
        m_ln = Marker(
            header=header,
            ns=f"ln_{name}",
            id=1,
            type=Marker.LINE_LIST,
        )
        m_ln.scale.x = 0.02
        m_ln.color.b = 1.0; m_ln.color.a = 1.0
        for i,j in self.skel_conns:
            if i in pts3d and j in pts3d:
                m_ln.points.append(Point(x=pts3d[i][0], y=pts3d[i][1], z=pts3d[i][2]))
                m_ln.points.append(Point(x=pts3d[j][0], y=pts3d[j][1], z=pts3d[j][2]))
        markers.markers.append(m_ln)

        return markers

def main(args=None):
    rclpy.init(args=args)
    node = MoveNet3DNode()
    executor = SingleThreadedExecutor()  # Usamos un solo hilo para evitar problemas de sincronización
    # executor = MultiThreadedExecutor()  # Alternativa si se necesita concurrencia
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down MoveNet3DNode...')
    finally:
        node.destroy_node()
        rclpy.shutdown()
