import mediapipe as mp
import cv2
from .abstract_pose_detector import PoseDetector

# --- Mediapipe Implementation ---
class MediapipePoseDetector(PoseDetector):
    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1, enable_segmentation: bool = False):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation
        )

    def process(self, image: cv2.Mat):
        # Expect RGB image
        return self.pose.process(image).pose_landmarks
