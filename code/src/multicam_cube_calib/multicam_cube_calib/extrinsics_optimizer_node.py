#!/usr/bin/env python3
import rclpy, numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile
from multicam_cube_calib_interfaces.msg import PairMeasurement
from std_srvs.srv import Trigger
from std_srvs.srv import Trigger
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from collections import deque, defaultdict
from multicam_cube_calib.se3 import tf_to_mat, mat_to_tf, so3_log, so3_exp
import yaml, os, time

class ExtrinsicsOptimizer(Node):
    def __init__(self):
        super().__init__('extrinsics_optimizer')
        self.declare_parameter('root_camera', 'cam00/camera_00')
        self.declare_parameter('cameras', ['cam00/camera_00','cam00/camera_01','cam01/camera_02', 'cam02/camera_03'])
        self.declare_parameter('max_pairs', 1000)
        self.declare_parameter('optimize_rate_hz', 5.0)
        self.declare_parameter('use_huber', True)
        self.declare_parameter('huber_delta', 0.1)

        self.root = self.get_parameter('root_camera').get_parameter_value().string_value
        self.cams = list(self.get_parameter('cameras').get_parameter_value().string_array_value)
        self.max_pairs = int(self.get_parameter('max_pairs').get_parameter_value().integer_value)
        self.use_huber = bool(self.get_parameter('use_huber').get_parameter_value().bool_value)
        self.huber_delta = float(self.get_parameter('huber_delta').get_parameter_value().double_value)
        self.optimize_dt = 1.0/float(self.get_parameter('optimize_rate_hz').get_parameter_value().double_value)

        # Estado: transform cam->root (root = identidad)
        self.cam_index = {c:i for i,c in enumerate(self.cams)}
        self.X = [np.eye(4) for _ in self.cams]  # X[k]: T_root->cam_k (publicaremos root->cam)
        # Medidas acumuladas (ventana)
        self.pairs = deque(maxlen=self.max_pairs)

        self.running = False

        self.sub = self.create_subscription(PairMeasurement, '/calib/pairs', self.on_pair, QoSProfile(depth=50))
        self.broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(self.optimize_dt, self.on_timer)

        # Servicios
        self.srv_start = self.create_service(Trigger, 'start', self.srv_start_cb)
        self.srv_stop  = self.create_service(Trigger, 'stop',  self.srv_stop_cb)
        self.srv_reset = self.create_service(Trigger, 'reset', self.srv_reset_cb)
        self.srv_save  = self.create_service(Trigger, 'save', self.srv_save_cb)

        self.get_logger().info(f"Optimizer listo. Root={self.root}. Cámaras={self.cams}")

    # ====== Servicios ======
    def srv_start_cb(self, req, res):
        self.running = True
        res.success = True
        res.message = 'Calibración en marcha'
        self.get_logger().info("Calibración iniciada")
        return res

    def srv_stop_cb(self, req, res):
        self.running = False
        res.success = True
        res.message = 'Calibración detenida'
        return res

    def srv_reset_cb(self, req, res):
        self.pairs.clear()
        self.X = [np.eye(4) for _ in self.cams]
        self.running = False
        res.success = True
        res.message = 'Estado reseteado'
        return res

    def srv_save_cb(self, req, res):
        try:
            path = req.filepath if req.filepath else f'/tmp/extrinsics_{int(time.time())}.yaml'
            data = {'root': self.root, 'cameras': {}}
            for c, Xc in zip(self.cams, self.X):
                # Guardamos root->cam como [qx,qy,qz,qw, tx,ty,tz]
                from multicam_cube_calib.se3 import mat_to_quat_trans
                q, t = mat_to_quat_trans(Xc)
                data['cameras'][c] = {'q': list(q), 't': list(t)}
            with open(path, 'w') as f:
                yaml.safe_dump(data, f)
            res.success = True
            res.message = f'Guardado en {path}'
        except Exception as e:
            res.success = False
            res.message = f'Error guardando: {e}'
        return res

    # ====== Sub y timer ======
    def on_pair(self, msg: PairMeasurement):
        self.get_logger().debug(f"Recibido par {msg.cam_i} -> {msg.cam_j} con peso {msg.weight:.4f}")
        # Convertir a matriz
        Tij = tf_to_mat(msg.t_i_to_j)
        wi = float(max(1e-6, msg.weight))
        i = self.cam_index.get(msg.cam_i, None)
        j = self.cam_index.get(msg.cam_j, None)
        if i is None or j is None:
            return
        self.pairs.append((i, j, Tij, wi))

    def on_timer(self):
        self.get_logger().info(f"Timer de optimización activado, pares acumulados: {len(self.pairs)}")
        # Publica TF siempre con el último estado
        self.publish_tf()
        if not self.running or len(self.pairs) < 3:
            return
        self.get_logger().info(f'Optimizando con {len(self.pairs)} pares...')
        # Hacer 2-3 iteraciones GN sobre ventana
        for _ in range(3):
            H, b = self.build_normal_equations()
            # Fijar root: su bloque no se optimiza (X[root]=I). Ya que root podría no ser índice 0, manejamos al construir H.
            # Resolver
            try:
                delta = np.linalg.lstsq(H, -b, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            # Aplicar actualización pequeña
            self.apply_delta(delta)
            # Parada si pequeño
            if np.linalg.norm(delta) < 1e-6:
                break

    # ====== Optimización ======
    def build_normal_equations(self):
        n = len(self.cams)
        # variables: todas excepto root -> (n-1)*6
        var_map = {}
        idx = 0
        for k,c in enumerate(self.cams):
            if c == self.root:
                continue
            var_map[k] = (idx, idx+6)
            idx += 6
        m = idx
        H = np.zeros((m,m))
        b = np.zeros(m)
        for (i, j, Z, w) in list(self.pairs):
            # residual r = Log( Z^-1 * Xi^-1 * Xj )  -> approx: [t; log(R)]
            Xi = self.X[i]
            Xj = self.X[j]
            M = np.linalg.inv(Z) @ np.linalg.inv(Xi) @ Xj
            r = np.zeros(6)
            r[:3] = M[:3,3]
            r[3:] = so3_log(M[:3,:3])
            # robustez
            if self.use_huber:
                nl = np.linalg.norm(r)
                delta = self.huber_delta
                if nl <= delta:
                    rw = 1.0
                else:
                    rw = delta / nl
            else:
                rw = 1.0
            r = rw * w * r

            # Jacobianos numéricos sobre Xi y Xj (si son variables)
            eps = 1e-5
            def res_func(Xi_new, Xj_new):
                M2 = np.linalg.inv(Z) @ np.linalg.inv(Xi_new) @ Xj_new
                rr = np.zeros(6)
                rr[:3] = M2[:3,3]
                rr[3:] = so3_log(M2[:3,:3])
                return rr

            J_i = None
            J_j = None
            if i in var_map:
                J_i = np.zeros((6,6))
                for k in range(6):
                    d = np.zeros(6); d[k] = eps
                    Xi_p = Xi @ se3_inc(d)  # left inc sobre Xi
                    rp = res_func(Xi_p, Xj)
                    J_i[:,k] = (rp - r) / eps
                J_i = rw * w * J_i
            if j in var_map:
                J_j = np.zeros((6,6))
                for k in range(6):
                    d = np.zeros(6); d[k] = eps
                    Xj_p = Xj @ se3_inc(d)
                    rp = res_func(Xi, Xj_p)
                    J_j[:,k] = (rp - r) / eps
                J_j = rw * w * J_j

            # Acumular en H,b
            if i in var_map:
                a,b_i = var_map[i]
                ri = r.copy()
                if j in var_map:
                    a2,b_j = var_map[j]
                    H[a:b_i, a:b_i] += J_i.T @ J_i
                    H[a:b_i, a2:b_j] += J_i.T @ J_j
                    H[a2:b_j, a:b_i] += J_j.T @ J_i
                    H[a2:b_j, a2:b_j] += J_j.T @ J_j
                    b_vec = J_i.T @ ri
                    b[a:b_i] += b_vec
                    b[a2:b_j] += J_j.T @ ri
                else:
                    H[a:b_i, a:b_i] += J_i.T @ J_i
                    b[a:b_i] += J_i.T @ ri
            elif j in var_map:
                a2,b_j = var_map[j]
                H[a2:b_j, a2:b_j] += J_j.T @ J_j
                b[a2:b_j] += J_j.T @ r
        # Damping leve para estabilidad
        H += 1e-6*np.eye(H.shape[0])
        return H, b

    def apply_delta(self, delta):
        # Aplicación por bloques de 6
        var_blocks = []
        for k,c in enumerate(self.cams):
            if c == self.root:
                var_blocks.append(None)
                continue
            var_blocks.append(k)
        ptr = 0
        for k,c in enumerate(self.cams):
            if c == self.root:
                continue
            d = delta[ptr:ptr+6]
            ptr += 6
            self.X[k] = self.X[k] @ se3_inc(d)

    def publish_tf(self):
        # Publicar root->cam TF dinámico
        now = self.get_clock().now().to_msg()
        for c, Xc in zip(self.cams, self.X):
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self.root
            t.child_frame_id = c
            from multicam_cube_calib.se3 import mat_to_tf
            t.transform = mat_to_tf(Xc)
            self.broadcaster.sendTransform(t)

# Helpers SE3 incrementales
from multicam_cube_calib.se3 import so3_exp

def se3_inc(d):
    v = d[:3]; w = d[3:]
    T = np.eye(4)
    T[:3,:3] = so3_exp(w)
    T[:3,3] = v
    return T


def main():
    rclpy.init()
    node = ExtrinsicsOptimizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()