#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoSubscriber(Node):
    def __init__(self):
        super().__init__('aruco_subscriber')

        # --- Parámetros del usuario ---
        self.image_topic = '/camera/color/image_raw'
        self.camerainfo_topic = '/camera/color/camera_info'
        self.marker_size = 0.16  # metros (EJEMPLO: 5 cm). ¡Cámbialo a tu tamaño real!

        # --- Subscripciones ---
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )
        self.camerainfo_sub = self.create_subscription(
            CameraInfo, self.camerainfo_topic, self.camerainfo_callback, 10
        )

        # --- Utilidades ---
        self.bridge = CvBridge()
        self.camera_info_received = False
        self.camera_matrix = None
        self.dist_coeffs = None

        # --- ArUco ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters()  # Para OpenCV >= 4.7
        # Si usas OpenCV antiguo, cambia a:
        # self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.get_logger().info('ArucoSubscriber inicializado. Esperando CameraInfo...')

    # ---------- Callbacks ----------
    def camerainfo_callback(self, msg: CameraInfo):
        # Extraer K (3x3) y D (coef. distorsión) del CameraInfo
        try:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64).reshape(-1, 1)  # (N,1)
            self.camera_info_received = True
            # Solo para confirmar una vez
            self.get_logger().info(
                f'CameraInfo recibido. K=\n{self.camera_matrix}\nD={self.dist_coeffs.ravel()}'
            )
            # Podemos desuscribirnos si solo quieres la primera vez:
            # self.destroy_subscription(self.camerainfo_sub)
        except Exception as e:
            self.get_logger().error(f'Error procesando CameraInfo: {e}')
            self.camera_info_received = False

    def image_callback(self, msg: Image):
        # Convertir imagen ROS -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Detectar ArUco
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Solo estimamos pose si ya tenemos calibración
            if self.camera_info_received and self.marker_size is not None:
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners,
                        self.marker_size,
                        self.camera_matrix,
                        self.dist_coeffs
                    )

                    # Dibujar ejes y mostrar datos
                    axis_len = float(self.marker_size) * 0.5  # longitud visual de los ejes
                    for i in range(len(ids)):
                        # Ejes en el centro del ArUco (origen de su SR)
                        self.draw_axis(
                            frame,
                            self.camera_matrix,
                            self.dist_coeffs,
                            rvecs[i],
                            tvecs[i],
                            axis_len
                        )

                        # Mostrar texto con ID y tvec (posición del marcador en SR de la cámara)
                        t = tvecs[i][0]  # (x, y, z) en metros
                        # Coordenadas 2D para ubicar el texto (esquina sup-izq del marcador)
                        c = corners[i][0]
                        x_txt, y_txt = int(c[0][0]), int(c[0][1]) - 10

                        cv2.putText(
                            frame,
                            f'ID {int(ids[i])} | x={t[0]:.3f} y={t[1]:.3f} z={t[2]:.3f} m',
                            (x_txt, max(y_txt, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
                except Exception as e:
                    self.get_logger().warn(f'No se pudo estimar la pose: {e}')
            else:
                # Aviso visual si aún no hay CameraInfo
                cv2.putText(
                    frame,
                    'Esperando /camera_info para dibujar ejes...',
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

        # Mostrar imagen
        cv2.imshow("Aruco Pose (ejes en el centro)", frame)
        cv2.waitKey(1)

    def draw_axis(self, img, camera_matrix, dist_coeffs, rvec, tvec, length=0.03):
        # Ejes en el sistema del marcador (X, Y, Z)
        axis = np.float32([
            [length, 0, 0],   # eje X
            [0, length, 0],   # eje Y
            [0, 0, length],   # eje Z
        ]).reshape(-1, 3)

        # Origen (0,0,0)
        origin = np.float32([[0, 0, 0]])

        # Proyectar origen y ejes a la imagen
        origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, camera_matrix, dist_coeffs)
        axes_2d, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

        o = tuple(origin_2d.ravel().astype(int))
        x = tuple(axes_2d[0].ravel().astype(int))
        y = tuple(axes_2d[1].ravel().astype(int))
        z = tuple(axes_2d[2].ravel().astype(int))

        # Dibujar líneas (colores similares a OpenCV)
        cv2.line(img, o, x, (0, 0, 255), 2)   # X - rojo
        cv2.line(img, o, y, (0, 255, 0), 2)   # Y - verde
        cv2.line(img, o, z, (255, 0, 0), 2)   # Z - azul


def main(args=None):
    rclpy.init(args=args)
    node = ArucoSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
