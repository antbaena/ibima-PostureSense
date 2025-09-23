#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrador GUI para ROS 2 Jazzy (Python + PySide6)

Características:
- Ventana principal con título "Calibrador" y texto descriptivo (lorem ipsum temporal).
- Rejilla 2x2 con 4 vistas de cámara, cada una suscrita a un tópico de imagen de ROS 2.
- Si no hay señal, muestra: "No hay señal del topic <nombre>".
- Debajo de cada imagen se muestra el nombre del tópico.
- Botón rojo "Salir" (cierra la app) y botón verde "Iniciar calibración" (abre una ventana placeholder).
- Integración Qt <-> rclpy con QTimer, sin hilos adicionales.

Dependencias:
  pip install PySide6 opencv-python numpy
Y los paquetes de ROS 2 (Jazzy): rclpy, sensor_msgs, cv_bridge (instalación vía apt/rosdep).

Ejecución (en un entorno con ROS 2 sourced):
  python calibrador_gui.py \
    --topics /camera0/image_raw /camera1/image_raw /camera2/image_raw /camera3/image_raw

"""
from __future__ import annotations
import sys
import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QMainWindow,
    QFrame,
    QDialog,
    QSizePolicy,
)

# --- ROS 2 ---
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import Image as RosImage
    from cv_bridge import CvBridge
except Exception as e:
    # Permit import-time error message if ROS 2 libs are missing, but keep module importable for editing.
    rclpy = None
    Node = object  # type: ignore
    RosImage = object  # type: ignore
    CvBridge = None  # type: ignore


# -------------------- Utilidades de imagen --------------------

def cv2_to_qimage(cv_img: np.ndarray) -> QImage:
    """Convierte una imagen OpenCV (BGR o RGB/mono) a QImage."""
    if cv_img is None:
        return QImage()

    if len(cv_img.shape) == 2:
        h, w = cv_img.shape
        bytes_per_line = w
        qimg = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        return qimg.copy()

    # Si es BGR, convertir a RGB
    if cv_img.shape[2] == 3:
        rgb = cv_img[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return qimg.copy()

    # RGBA
    if cv_img.shape[2] == 4:
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qimg = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        return qimg.copy()

    return QImage()


def make_placeholder_pixmap(size: QSize, message: str) -> QPixmap:
    """Crea un QPixmap con un mensaje centrado para estados de 'sin señal'."""
    w = max(size.width(), 320)
    h = max(size.height(), 240)
    pix = QPixmap(w, h)
    pix.fill(Qt.black)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.white)
    font = QFont()
    font.setPointSize(12)
    font.setBold(True)
    painter.setFont(font)
    rect = pix.rect()
    painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, message)
    painter.end()
    return pix


# -------------------- Widgets --------------------

class CameraTile(QFrame):
    def __init__(self, topic: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.topic = topic
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("cameraTile")
        self.setStyleSheet(
            """
            QFrame#cameraTile { border: 1px solid #2a2a2a; border-radius: 12px; }
            QLabel#topicLabel { color: #888; font-size: 12px; }
            """
        )

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(240, 180)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setObjectName("imageCanvas")

        self.topic_label = QLabel(self.topic)
        self.topic_label.setObjectName("topicLabel")
        self.topic_label.setAlignment(Qt.AlignCenter)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        lay.addWidget(self.image_label, 1)
        lay.addWidget(self.topic_label, 0)

        # Estado inicial (sin señal)
        self.set_no_signal()

    def set_frame(self, qimg: QImage):
        if qimg.isNull():
            self.set_no_signal()
            return
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def set_no_signal(self):
        msg = f"No hay señal del topic\n{self.topic}"
        placeholder = make_placeholder_pixmap(self.image_label.size(), msg)
        self.image_label.setPixmap(placeholder)

    def resizeEvent(self, event):
        # Re-pintar placeholder a nuevo tamaño para que no quede pixelado
        if self.image_label.pixmap() and self.image_label.pixmap().isNull() is False:
            # Si el placeholder estaba puesto, lo regeneramos
            # Heurística: si el pixmap actual es totalmente negro con texto,
            # no tenemos forma simple de detectarlo; por simplicidad, refrescamos con el mismo contenido.
            self.set_no_signal()
        super().resizeEvent(event)


class CalibracionDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Calibración (WIP)")
        self.setMinimumSize(480, 320)
        lbl = QLabel("Aquí irá el flujo de calibración. \n(Placeholder – por ahora sin lógica).")
        lbl.setAlignment(Qt.AlignCenter)
        lay = QVBoxLayout(self)
        lay.addWidget(lbl)


class MainWindow(QMainWindow):
    def __init__(self, topics: List[str]):
        super().__init__()
        self.setWindowTitle("Calibrador")
        self.setMinimumSize(1100, 800)

        container = QWidget()
        self.setCentralWidget(container)

        title = QLabel("Calibrador")
        title.setAlignment(Qt.AlignCenter)
        tfont = QFont()
        tfont.setPointSize(22)
        tfont.setBold(True)
        title.setFont(tfont)

        desc = QLabel(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Suspendisse potenti. Integer at velit ac dui consequat tempus. "
            "Mauris non lorem sit amet mi facilisis cursus."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)

        # Rejilla de 2x2
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        self.tiles: List[CameraTile] = []
        for i, topic in enumerate(topics):
            tile = CameraTile(topic)
            self.tiles.append(tile)
            r = i // 2
            c = i % 2
            grid.addWidget(tile, r, c)

        # Botones
        btn_salir = QPushButton("Salir")
        btn_salir.setStyleSheet("background:#d32f2f; color:white; padding:10px; border-radius:10px; font-weight:bold;")
        btn_salir.clicked.connect(self.close)

        btn_iniciar = QPushButton("Iniciar calibración")
        btn_iniciar.setStyleSheet("background:#2e7d32; color:white; padding:10px; border-radius:10px; font-weight:bold;")
        btn_iniciar.clicked.connect(self.open_calibration)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(btn_salir)
        buttons.addWidget(btn_iniciar)
        buttons.addStretch(1)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(grid, 1)
        layout.addLayout(buttons)

        # ROS hookup
        self.ros: Optional[RosBridge] = None
        if rclpy is None:
            print("[ADVERTENCIA] ROS 2 (rclpy) no está disponible en este entorno. La GUI abrirá sin señal.")
        else:
            self.ros = RosBridge(topics)

        # Timer para refrescar frames
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(33)  # ~30 FPS
        self.refresh_timer.timeout.connect(self.on_refresh)
        self.refresh_timer.start()

    def on_refresh(self):
        # Integración con el executor de rclpy, si existe
        if self.ros is not None:
            self.ros.spin_once(0.0)

        # Actualizar frames en tiles
        if self.ros is None:
            return
        for tile in self.tiles:
            qimg, alive = self.ros.get_qimage(tile.topic)
            if qimg is not None and alive:
                tile.set_frame(qimg)
            elif not alive:
                tile.set_no_signal()

    def open_calibration(self):
        dlg = CalibracionDialog(self)
        dlg.exec()

    def closeEvent(self, event):
        if self.ros is not None:
            self.ros.shutdown()
        super().closeEvent(event)


# -------------------- ROS Bridge --------------------

class RosBridge(Node):
    """Nodo ROS que gestiona 4 suscripciones de imagen y expone frames como QImage."""

    def __init__(self, topics: List[str]):
        if rclpy is None:
            raise RuntimeError("ROS2 no inicializado")

        # Inicializar rclpy una única vez
        if not rclpy.ok():
            rclpy.init(args=None)
        super().__init__("calibrador_gui")

        self.bridge = CvBridge() if CvBridge is not None else None
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self)

        self.topics = topics
        self._latest_frames: Dict[str, Tuple[Optional[QImage], float]] = {t: (None, 0.0) for t in topics}

        self._subs = []
        for t in topics:
            sub = self.create_subscription(RosImage, t, self._on_image_factory(t), 10)
            self._subs.append(sub)

        # Timeout para considerar "sin señal"
        self.dead_timeout_sec = 1.5

    # ----- API pública
    def spin_once(self, timeout_sec: float = 0.0):
        try:
            self.executor.spin_once(timeout_sec=timeout_sec)
        except Exception:
            pass

    def get_qimage(self, topic: str) -> Tuple[Optional[QImage], bool]:
        qimg, ts = self._latest_frames.get(topic, (None, 0.0))
        alive = (time.time() - ts) < self.dead_timeout_sec if ts else False
        return qimg, alive

    def shutdown(self):
        try:
            self.executor.remove_node(self)
            self.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    # ----- Callbacks
    def _on_image_factory(self, topic: str):
        def _cb(msg: RosImage):
            qimg = self._rosimg_to_qimage(msg)
            self._latest_frames[topic] = (qimg, time.time())
        return _cb

    def _rosimg_to_qimage(self, msg: RosImage) -> Optional[QImage]:
        try:
            if self.bridge is None:
                return None
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # Normalizar formatos comunes a RGB/mono
            if cv_img.ndim == 2:
                pass  # gris
            elif cv_img.shape[2] == 3:
                # En ROS suele ser BGR8 o RGB8
                # Si encoding indica 'rgb8', ya vendría en RGB; si es 'bgr8', convertimos en cv2_to_qimage
                pass
            elif cv_img.shape[2] == 4:
                pass
            qimg = cv2_to_qimage(cv_img)
            return qimg
        except Exception:
            return None


# -------------------- main --------------------

def parse_args(argv: List[str]) -> List[str]:
    parser = argparse.ArgumentParser(description="Calibrador GUI para ROS 2 Jazzy")
    parser.add_argument(
        "--topics",
        nargs=4,
        metavar=("TOPIC0", "TOPIC1", "TOPIC2", "TOPIC3"),
        default=[
            "/camera0/image_raw",
            "/camera1/image_raw",
            "/camera2/image_raw",
            "/camera3/image_raw",
        ],
        help="Cuatro tópicos de imagen sensor_msgs/msg/Image (por defecto: /camera{0..3}/image_raw)",
    )
    args = parser.parse_args(argv)
    return args.topics


def main(argv: Optional[List[str]] = None):
    argv = argv if argv is not None else sys.argv[1:]
    topics = parse_args(argv)

    app = QApplication(sys.argv)
    win = MainWindow(topics)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
