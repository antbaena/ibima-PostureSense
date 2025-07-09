import time
import numpy as np
import cv2

# Implementación de la clase Marker con filtrado y scoring
class Marker:
    def __init__(self, marker_id: int, marker_length: float, timeout: float = 5.0, filter_alpha: float = 0.2):
        self.id = marker_id
        self.marker_length = marker_length
        self.timeout = timeout
        self.first_seen = None
        self.last_seen = None
        self.reliable = False
        self.score = 0.0
        self.filtered_translation = None
        self.filtered_rotation = None  # Se guarda como vector de rotación
        self.filter_alpha = filter_alpha
        self.variability = 0.0  # Medida de la variabilidad en las actualizaciones
        self.reliable_transform = None  # Nueva propiedad para almacenar la transformación filtrada


    def update(self, translation_matrix: np.ndarray):
        current_time = time.time()
        if self.first_seen is None:
            self.first_seen = current_time
        self.last_seen = current_time

        # Extraer la traslación y la rotación de la matriz 4x4
        new_translation = translation_matrix[:3, 3]
        new_rot_matrix = translation_matrix[:3, :3]
        new_rvec, _ = cv2.Rodrigues(new_rot_matrix)
        new_rvec = new_rvec.flatten()

        if self.filtered_translation is None:
            # Primera actualización: inicializar los valores filtrados
            self.filtered_translation = new_translation
            self.filtered_rotation = new_rvec
            diff_translation = 0.0
            diff_rotation = 0.0
        else:
            # Calcular la diferencia (error) entre la medición y el valor filtrado
            diff_translation = np.linalg.norm(new_translation - self.filtered_translation)
            diff_rotation = np.linalg.norm(new_rvec - self.filtered_rotation)
            # Actualización del filtro (exponential smoothing)
            self.filtered_translation = self.filter_alpha * new_translation + (1 - self.filter_alpha) * self.filtered_translation
            self.filtered_rotation = self.filter_alpha * new_rvec + (1 - self.filter_alpha) * self.filtered_rotation

        diff_total = diff_translation + diff_rotation

        # Actualizar la variabilidad como una media móvil exponencial del error total
        self.variability = self.filter_alpha * diff_total + (1 - self.filter_alpha) * self.variability

        # Sistema de scoring: si la diferencia total es pequeña, se incrementa el score;
        # de lo contrario, se decrementa (sin caer por debajo de 0)
        threshold = 0.1  # Umbral arbitrario para considerar una actualización estable
        if diff_total < threshold:
            self.score += 1
        else:
            self.score = max(0, self.score - 1)

        # Construir la transformación filtrada (matriz homogénea 4x4)
        rot_matrix_filtered, _ = cv2.Rodrigues(self.filtered_rotation)
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix_filtered
        transform[:3, 3] = self.filtered_translation
        self.reliable_transform = transform


    def is_timed_out(self):
        if self.last_seen is None:
            return False
        return (time.time() - self.last_seen) > self.timeout

    def is_precise(self):
        # Se considera que el marcador es preciso (y por ende reliable) cuando su score supera un umbral
        return self.score >= 50  # Este umbral se puede ajustar según pruebas

