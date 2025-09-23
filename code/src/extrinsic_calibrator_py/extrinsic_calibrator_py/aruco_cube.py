import yaml
import numpy as np
import tf_transformations as tft
import pyvista as pv
import os
import pyvista as pv
from pyvista import Plane
from PIL import Image

class ArucoCube:
    """
    Representación de un cubo perfecto con marcadores ArUco en cada cara.
    Permite obtener las transformaciones homogéneas entre cualquier par de marcadores,
    partiendo de configuraciones predeterminadas.

    Cada marcador se identifica por un ID único.
    """

    def __init__(self, marker_length: float, cube_side: float, marker_ids: dict = None):
        """
        :param marker_length: Lado del marcador ArUco (m).
        :param cube_side: Longitud del lado del cubo (m).
        :param marker_ids: Diccionario opcional mapping de caras a IDs, p.ej:
               {"+X": 0, "-X": 1, "+Y": 2, "-Y": 3, "+Z": 4, "-Z": 5}
        """
        self.marker_length = marker_length
        self.cube_side = cube_side
        # IDs por defecto si no se pasan
        default = {"+X": 0, "-X": 1, "+Y": 2, "-Y": 3, "+Z": 4, "-Z": 5}
        self.marker_ids = marker_ids or default
        # Precalcular poses de cada marcador en el sistema de coordenadas del cubo
        self._poses = self._compute_face_poses()

    def _compute_face_poses(self) -> dict:
        """
        Genera un dict {marker_id: 4x4 np.ndarray} con T_cube_to_marker.
        Coloca cada marcador en el centro de cada cara, orientado hacia afuera.
        """
        half = self.cube_side / 2.0
        poses = {}
        # Orientaciones de cara: cada tupla es (axis, direction)
        faces = {
            "+X": ([1, 0, 0],  1),  # cara derecha
            "-X": ([1, 0, 0], -1),  # cara izquierda
            "+Y": ([0, 1, 0],  1),  # cara adelante
            "-Y": ([0, 1, 0], -1),  # cara atrás
            "+Z": ([0, 0, 1],  1),  # cara arriba
            "-Z": ([0, 0, 1], -1),  # cara abajo
        }
        for face, (axis, sign) in faces.items():
            mid = self.marker_ids[face]
            # Posición: centro de la cara
            translation = [0, 0, 0]
            for idx, a in enumerate(axis):
                translation[idx] = sign * half * a
            # Rotación: el eje Z del marcador mira hacia el exterior del cubo
            # Construimos vector objetivo (eje Z_local) = axis * sign
            target_z = np.array(axis, dtype=float) * sign
            # Elegimos un up arbitrario, distinto de parallelismo
            up_guess = np.array([0, 0, 1], dtype=float) if abs(target_z.dot([0,0,1])) < 0.99 else np.array([0,1,0], dtype=float)
            # X_local = up_guess cross target_z
            x_axis = np.cross(up_guess, target_z)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(target_z, x_axis)
            # Montamos rot matrix
            R = np.stack((x_axis, y_axis, target_z), axis=1)
            # Homogénea
            T = np.eye(4)
            T[:3,:3] = R
            T[:3, 3] = translation
            poses[mid] = T
        return poses

    def get_transform(self, id_from: int, id_to: int) -> np.ndarray:
        """
        Devuelve la matriz homogénea 4x4 que transforma coordenadas de marcador id_from a id_to.
        """
        if id_from not in self._poses or id_to not in self._poses:
            raise KeyError(f"IDs desconocidos: {id_from}, {id_to}")
        T_cube_from = self._poses[id_from]
        T_cube_to   = self._poses[id_to]
        # T_from_to = inv(T_cube_from) * T_cube_to
        return np.linalg.inv(T_cube_from) @ T_cube_to

    @staticmethod
    def load_from_yaml(path: str) -> 'ArucoCube':
        """
        Carga configuración de YAML con keys:
          marker_length: float
          cube_side: float
          marker_ids: {"+X": id0, ...}
        """
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        return ArucoCube(
            marker_length=cfg['marker_length'],
            cube_side=cfg['cube_side'],
            marker_ids=cfg.get('marker_ids', None)
        )

    def save_to_yaml(self, path: str) -> None:
        """
        Guarda la configuración actual en YAML.
        """
        data = {
            'marker_length': self.marker_length,
            'cube_side': self.cube_side,
            'marker_ids': self.marker_ids
        }
        with open(path, 'w') as f:
            yaml.safe_dump(data, f)

    def visualize_cube_with_textures_and_labels(self, aruco_image_dir: str):
        """
        Visualiza el cubo con texturas de marcadores ArUco y texto con el ID y la orientación.

        :param aruco_image_dir: Carpeta que contiene las imágenes de los marcadores.
        """
        plotter = pv.Plotter()
        cube = pv.Cube(center=(0, 0, 0), x_length=self.cube_side, y_length=self.cube_side, z_length=self.cube_side)
        
        # Cubo base semitransparente
        plotter.add_mesh(cube, color='lightgray', opacity=0.2, show_edges=True)

        half = self.cube_side / 2.0
        face_data = {
            "+X": (( half, 0, 0), (1, 0, 0), "right"),
            "-X": ((-half, 0, 0), (-1, 0, 0), "left"),
            "+Y": ((0,  half, 0), (0, 1, 0), "front"),
            "-Y": ((0, -half, 0), (0, -1, 0), "back"),
            "+Z": ((0, 0,  half), (0, 0, 1), "top"),
            "-Z": ((0, 0, -half), (0, 0, -1), "bottom"),
        }

        for face, (center, normal, face_name) in face_data.items():
            marker_id = self.marker_ids.get(face, '?')
            image_path = os.path.join(aruco_image_dir, f"id{marker_id}.png")
            if not os.path.isfile(image_path):
                print(f"[AVISO] No se encuentra la imagen: {image_path}")
                continue

            # Crear plano orientado correctamente
            plane = pv.Plane(center=center,
                            direction=normal,
                            i_size=self.marker_length,
                            j_size=self.marker_length)
            
            # Cargar textura
            texture = pv.read_texture(image_path)
            plotter.add_mesh(plane, texture=texture)

            # Posición de etiqueta flotante
            label_offset = 0.015
            label_pos = tuple(np.array(center) + label_offset * np.array(normal))
            label_text = f"{face_name}\nID: {marker_id}"

            plotter.add_point_labels(
                [label_pos],
                [label_text],
                text_color='black',
                font_size=12,
                point_color='white',
                point_size=20
            )

        plotter.show_grid()
        plotter.set_background("white")
        plotter.show(title="ArucoCube con Texturas y Etiquetas")

if __name__ == "__main__":
    # Ejemplo de uso
    marker_length = 0.04  # 4 cm
    cube_side = 0.10      # 10 cm
    marker_ids = {"+X": 0, "-X": 1, "+Y": 2, "-Y": 3, "+Z": 4, "-Z": 5}

    cube = ArucoCube(marker_length, cube_side, marker_ids)

    # Mostrar transformaciones entre dos caras
    T = cube.get_transform(0, 2)  # De +X a +Y
    print("Transformación de +X a +Y:\n", T)

    # Visualizar el cubo
    cube.visualize_cube_with_textures_and_labels("/home/ubuntu/ibima-PostureSense/code/src/extrinsic_calibrator_py/extrinsic_calibrator_py/arucos_png")