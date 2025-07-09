import numpy as np
from prettytable import PrettyTable

def compose_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    """Dado un listado de T_ij, devuelve una única matriz fiable por pseudoinversa."""
    # Ejemplo: stack horizontal + pinv + identity stack
    # …

def find_paths_between_markers(adj_matrix: list[list[bool]]) -> list[list[int]]:
    """Encuentra todos los caminos en grafo de marcadores."""
    # Algoritmo BFS/DFS…
    # …

def display_camera_to_marker_table(title, table_data, array_of_cameras, logger):
    table = PrettyTable()
    table.field_names = ["Marker ID"] + [f"Camera {i}" for i in range(len(array_of_cameras))]
    if type(table_data[0][0]) is bool:
        for marker_id, row in enumerate(table_data):
            table.add_row([marker_id] + ['✓' if cell else '✗' for cell in row])
    else:
        for marker_id, row in enumerate(table_data):
            table.add_row([marker_id] + [cell for cell in row])
    logger.info(f"{title}\n" + table.get_string())

def display_marker_to_marker_table(title, table_data, largest_marker, logger):
        table = PrettyTable()
        table.field_names = ["Marker ID"] + [f"Marker {i}" for i in range(largest_marker + 1)]
        if type(table_data[0][0]) is bool:
            for marker_id, row in enumerate(table_data):
                table.add_row([marker_id] + ['✓' if cell else '✗' for cell in row])
        else:
            for marker_id, row in enumerate(table_data):
                table.add_row([marker_id] + [cell for cell in row])
        logger.info(f"{title}\n" + table.get_string())


