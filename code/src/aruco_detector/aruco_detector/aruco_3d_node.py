#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Point
from std_msgs.msg import Header

class Aruco3DNode(Node):
    def __init__(self):
        super().__init__('aruco_3d_detector')
        self.bridge = CvBridge()

        self.declare_parameter('color_image_topic', '/cam00/camera_00/color/image_raw/decompressed')
        self.declare_parameter('camera_info_topic', '/cam00/camera_00/color/camera_info')
        self.declare_parameter('camera_name', 'cam00')

        color_image_topic = self.get_parameter('color_image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        camera_name = self.get_parameter('camera_name').get_parameter_value().string_value

        # Suscriptores sincronizados
        img_sub = Subscriber(self, Image, color_image_topic)
        info_sub = Subscriber(self, CameraInfo, camera_info_topic)
        self.sync = ApproximateTimeSynchronizer([img_sub, info_sub],
                                                queue_size=10,
                                                slop=0.1)
        self.sync.registerCallback(self.callback)

        # Publicador de markers
        self.marker_pub = self.create_publisher(MarkerArray, camera_name + '/aruco_markers', 10)

        # Parámetros ArUco
        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Tamaño real de un marcador (en metros)
        self.marker_length = 0.26
        self.get_logger().info('Aruco 3D Detector Node initialized')

    def callback(self, img_msg: Image, info_msg: CameraInfo):
        # Convertir a OpenCV
        frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cam_matrix = np.array(info_msg.k).reshape((3,3))
        dist_coeffs = np.array(info_msg.d)

        # Detectar esquinas e IDs
        corners, ids, _ = cv2.aruco.detectMarkers(frame,
                                                  self.aruco_dict,
                                                  parameters=self.aruco_params)
        marker_array = MarkerArray()


        self.get_logger().info(f"Detectados {len(corners)} marcadores ArUco: {ids.flatten() if ids is not None else 'Ninguno'}")

        if ids is not None:
            for i, corner in enumerate(corners):
                # Definir puntos 3D del marcador (centrado en origen, en z=0)
                s = self.marker_length / 2.0
                obj_pts = np.array([
                    [-s,  s, 0],
                    [ s,  s, 0],
                    [ s, -s, 0],
                    [-s, -s, 0],
                ], dtype=np.float32)

                img_pts = corner.reshape(-1,2).astype(np.float32)

                # Calcular pose con RANSAC
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, cam_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                
            
                # Proyectar obj_pts a coordenadas de cámara
                R, _ = cv2.Rodrigues(rvec)
                pts_cam = (R @ obj_pts.T + tvec).T  # Nx3
                                # Color según ID
                r = ((ids[i][0] * 37) % 255) / 255.0
                g = ((ids[i][0] * 73) % 255) / 255.0
                b = ((ids[i][0] * 129) % 255) / 255.0

                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "map"


                # ---------- TEXTO CON EL ID DEL ARUCO ----------
                text_marker = Marker()
                text_marker.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
                text_marker.header = header
                text_marker.ns = f"aruco_id"
                text_marker.id = int(ids[i]) * 10 + 2
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.scale.z = 0.3  # tamaño del texto (alto de letra)
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                text_marker.text = str(ids[i][0])

                # Posicionar el texto justo encima del centro del marcador
                center = np.mean(pts_cam, axis=0)
                text_marker.pose.position.x = float(center[0])
                text_marker.pose.position.y = float(center[1])
                text_marker.pose.position.z = float(center[2]) + 0.05  # un poco arriba

                marker_array.markers.append(text_marker)



                # ---------- PUNTOS COMO ESFERAS (bolitas) ----------
                spheres = Marker()
                spheres.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
                spheres.header = header
                spheres.ns = f"aruco_spheres"
                spheres.id = int(ids[i]) * 10
                spheres.type = Marker.SPHERE_LIST
                spheres.action = Marker.ADD
                spheres.scale.x = 0.015  # diámetro de la bolita
                spheres.scale.y = 0.015
                spheres.scale.z = 0.015
                spheres.color.r = r
                spheres.color.g = g
                spheres.color.b = b
                spheres.color.a = 1.0

                # ---------- LÍNEAS ENTRE PUNTOS ----------
                lines = Marker()
                lines.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
                lines.header = header
                lines.ns = f"aruco_lines"
                lines.id = int(ids[i]) * 10 + 1
                lines.type = Marker.LINE_STRIP  # línea continua
                lines.action = Marker.ADD
                lines.scale.x = 0.005
                lines.color.r = r
                lines.color.g = g
                lines.color.b = b
                lines.color.a = 1.0

                # Añadir puntos a ambos markers
                for p in pts_cam:
                    pt = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                    spheres.points.append(pt)
                    lines.points.append(pt)

                # Cerrar el cuadrado (conectando el último con el primero)
                lines.points.append(lines.points[0])

                # Agregar ambos al array
                marker_array.markers.append(spheres)
                marker_array.markers.append(lines)

        # Publicar todos los markers encontrados
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = Aruco3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
