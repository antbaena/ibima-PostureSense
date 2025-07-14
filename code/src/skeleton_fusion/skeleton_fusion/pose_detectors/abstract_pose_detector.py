# detectors/pose_detector.py
from abc import ABC, abstractmethod
import cv2
import mediapipe as mp

class PoseDetector(ABC):
    @abstractmethod
    def process(self, image: cv2.Mat):
        """Recibe imagen RGB y devuelve lista de landmarks normalizados."""
        pass
