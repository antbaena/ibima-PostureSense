import numpy as np
from geometry_msgs.msg import Transform

# --- SO(3) helpers ---

def skew(w):
    x,y,z = w
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=float)


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)


def so3_log(R):
    cos_th = (np.trace(R)-1)/2.0
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-12:
        return np.zeros(3)
    w_hat = (R - R.T)/(2*np.sin(th))
    return np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]]) * th

# --- SE(3) conversions ---

def mat_to_tf(M):
    T = Transform()
    T.translation.x = float(M[0,3])
    T.translation.y = float(M[1,3])
    T.translation.z = float(M[2,3])
    q = R_to_quat(M[:3,:3])
    T.rotation.x, T.rotation.y, T.rotation.z, T.rotation.w = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    return T


def tf_to_mat(T: Transform):
    M = np.eye(4)
    M[:3,3] = [T.translation.x, T.translation.y, T.translation.z]
    R = quat_to_R([T.rotation.x, T.rotation.y, T.rotation.z, T.rotation.w])
    M[:3,:3] = R
    return M


def R_to_quat(R):
    # Returns [x,y,z,w]
    m00,m01,m02 = R[0]
    m10,m11,m12 = R[1]
    m20,m21,m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = (tr+1.0)**0.5 * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = (1.0 + m00 - m11 - m22)**0.5 * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = (1.0 + m11 - m00 - m22)**0.5 * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = (1.0 + m22 - m00 - m11)**0.5 * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qx,qy,qz, qw])


def quat_to_R(q):
    x,y,z,w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)


def mat_to_quat_trans(M):
    q = R_to_quat(M[:3,:3])
    t = M[:3,3]
    return q, t
