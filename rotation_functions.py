import numpy as np


def _normalize(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError('Rotation axis cannot be the zero vector.')
    return vector / norm


def rotate_vector(vector, axis, theta_deg):
    """Rotate a vector about an axis by theta_deg using Rodrigues' formula."""
    axis = _normalize(axis)
    vector = np.asarray(vector, dtype=float)
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c

    rotation_matrix = np.array([
        [t * axis[0] ** 2 + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
        [t * axis[0] * axis[1] + s * axis[2], t * axis[1] ** 2 + c, t * axis[1] * axis[2] - s * axis[0]],
        [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] ** 2 + c],
    ])
    return rotation_matrix @ vector


def rotate_two_vectors(vector, theta_1, theta_2, axis_1, axis_2):
    """Apply two ordered rotations, carrying the second axis with the first rotation."""
    vector_1 = rotate_vector(vector, axis_1, theta_1)
    axis_2_rotated = rotate_vector(axis_2, axis_1, theta_1)
    return rotate_vector(vector_1, axis_2_rotated, theta_2)


def rotate_three_vectors(vector, theta_1, theta_2, theta_3, axis_1, axis_2, axis_3):
    """Apply three ordered rotations, carrying downstream axes forward each step."""
    vector_1 = rotate_vector(vector, axis_1, theta_1)
    axis_2_rotated = rotate_vector(axis_2, axis_1, theta_1)
    axis_3_rotated = rotate_vector(axis_3, axis_1, theta_1)

    vector_2 = rotate_vector(vector_1, axis_2_rotated, theta_2)
    axis_3_rotated = rotate_vector(axis_3_rotated, axis_2_rotated, theta_2)

    return rotate_vector(vector_2, axis_3_rotated, theta_3)


def angle_between_vectors(v1, v2):
    """Return the angle in degrees between two vectors."""
    v1 = _normalize(v1)
    v2 = _normalize(v2)
    cosine_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))
