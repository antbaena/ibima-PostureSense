import numpy as np
from prettytable import PrettyTable


def compose_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    """Return a robust transform from a list of transforms using pseudoinverse."""
    if not transforms:
        return np.eye(4)

    stacked = np.hstack(transforms)
    num_blocks = stacked.shape[1] // 4
    stacked_identity = np.hstack([np.eye(4) for _ in range(num_blocks)])
    return stacked_identity @ np.linalg.pinv(stacked)


def find_paths_between_markers(adj_matrix: list[list[bool]]) -> list[list[int]]:
    """Return all paths in a graph described by an adjacency matrix."""
    size = len(adj_matrix)
    paths = []

    def dfs(current, visited, path):
        visited[current] = True
        path.append(current)
        for neighbor, connected in enumerate(adj_matrix[current]):
            if connected:
                if not visited[neighbor]:
                    dfs(neighbor, visited.copy(), path.copy())
                else:
                    paths.append(path + [neighbor])

    for start in range(size):
        dfs(start, [False] * size, [])

    return paths


def display_camera_to_marker_table(title, table_data, array_of_cameras, logger):
    table = PrettyTable()
    table.field_names = ["Marker ID"] + \
        [f"Camera {i}" for i in range(len(array_of_cameras))]
    if type(table_data[0][0]) is bool:
        for marker_id, row in enumerate(table_data):
            table.add_row([marker_id] + ['✓' if cell else '✗' for cell in row])
    else:
        for marker_id, row in enumerate(table_data):
            table.add_row([marker_id] + [cell for cell in row])
    logger.info(f"{title}\n" + table.get_string())


def display_marker_to_marker_table(title, table_data, largest_marker, logger):
    table = PrettyTable()
    table.field_names = ["Marker ID"] + \
        [f"Marker {i}" for i in range(largest_marker + 1)]
    if type(table_data[0][0]) is bool:
        for marker_id, row in enumerate(table_data):
            table.add_row([marker_id] + ['✓' if cell else '✗' for cell in row])
    else:
        for marker_id, row in enumerate(table_data):
            table.add_row([marker_id] + [cell for cell in row])
    logger.info(f"{title}\n" + table.get_string())
