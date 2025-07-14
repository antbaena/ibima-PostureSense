import tkinter as tk
from tkinter import messagebox
import threading
import json
from geometry_msgs.msg import Vector3, Quaternion
from extrinsic_calibrator_interfaces.msg import CameraTransform
from extrinsic_calibrator_interfaces.srv import BroadcastPoses
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String


class CameraCalibratorGUI(Node):
    def __init__(self):
        super().__init__('camera_calibrator_gui')
        self.root = tk.Tk()
        self.root.title("Camera Pose Optimizer")
        self.root.geometry("700x600")
        self.root.configure(bg="#f5f5f5")

        self.start_client = self.create_client(Trigger, '/start_optimization')
        self.finalize_client = self.create_client(Trigger, '/finalize_result')
        self.tf_broadcaster_client = self.create_client(BroadcastPoses, '/broadcast_static_tfs')

        # Frames
        self.main_frame = tk.Frame(self.root, bg="#f5f5f5")
        self.final_frame = tk.Frame(self.root, bg="#f5f5f5")

        self._setup_main_frame()
        self._setup_final_frame()

        self.main_frame.pack(fill='both', expand=True)

        self.create_subscription(String, '/camera_pose_metrics', self.metrics_callback, 10)
        self.root.after(100, self.ros_spin_once)
        self.optimization_result = {}
    def ros_spin_once(self):
        rclpy.spin_once(self, timeout_sec=0.01)
        self.root.after(100, self.ros_spin_once)

    def _setup_main_frame(self):
        label = tk.Label(self.main_frame, text="Pulsa 'Iniciar Optimización' para comenzar", font=("Arial", 14), bg="#f5f5f5", pady=10)
        label.pack()

        self.start_button = tk.Button(self.main_frame, text="Iniciar Optimización", font=("Arial", 12),
                                      bg="#4CAF50", fg="white", padx=10, pady=5, command=self.start_optimization)
        self.start_button.pack(pady=10)

        self.result_box = tk.Text(self.main_frame, height=20, width=80, state='disabled', wrap='word', font=("Courier", 10))
        self.result_box.pack(padx=20, pady=10)

        self.finalize_button = tk.Button(self.main_frame, text="Finalizar Calibración", font=("Arial", 12),
                                         bg="#2196F3", fg="white", padx=10, pady=5, command=self.finalize_result)
        self.finalize_button.pack(pady=10)

    def _setup_final_frame(self):
        title = tk.Label(self.final_frame, text="Parámetros de Calibración", font=("Arial", 16), bg="#f5f5f5")
        title.pack(pady=10)

        self.param_text = tk.Text(self.final_frame, height=18, width=80, state='disabled', wrap='word', font=("Courier", 10))
        self.param_text.pack(padx=20, pady=10)

        button_frame = tk.Frame(self.final_frame, bg="#f5f5f5")
        button_frame.pack(pady=10)

        self.publish_button = tk.Button(button_frame, text="Publicar TF Estática", font=("Arial", 12),
                                        bg="#4CAF50", fg="white", padx=10, pady=5, command=self.publish_tf)
        self.publish_button.grid(row=0, column=0, padx=10)

        self.exit_button = tk.Button(button_frame, text="Cancelar y Cerrar", font=("Arial", 12),
                                     bg="#f44336", fg="white", padx=10, pady=5, command=self.root.quit)
        self.exit_button.grid(row=0, column=1, padx=10)

        self.restart_button = tk.Button(self.final_frame, text="Reiniciar Medición", font=("Arial", 12),
                                        bg="#FFC107", fg="black", padx=10, pady=5, command=self.restart)
        self.restart_button.pack(pady=15)

    def start_optimization(self):
        threading.Thread(target=self._call_start_service).start()

    def _call_start_service(self):
        if not self.start_client.wait_for_service(timeout_sec=2.0):
            self._update_result("Servicio no disponible (/start_optimization)")
            return

        req = Trigger.Request()
        future = self.start_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result().success:
            self._update_result("Optimización iniciada\n" + future.result().message)
        else:
            self._update_result("Error al iniciar: " + future.result().message)

    def finalize_result(self):
        threading.Thread(target=self._call_finalize_service).start()

    def _call_finalize_service(self):
        if not self.finalize_client.wait_for_service(timeout_sec=2.0):
            self._update_result("Servicio no disponible (/finalize_result)")
            return

        req = Trigger.Request()
        future = self.finalize_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result().success:
            try:
                data = json.loads(future.result().message)
                self.optimization_result = data
                self._switch_to_final_screen()
                self._populate_camera_cards(data)
            except Exception as e:
                self._update_result(f"Error al interpretar resultado: {str(e)}")
        else:
            self._update_result("Error al finalizar: " + future.result().message)


    def _switch_to_final_screen(self):
        self.main_frame.pack_forget()
        self.final_frame.pack(fill='both', expand=True)

    def _populate_camera_cards(self, camera_data):
        # Limpia frame
        for widget in self.final_frame.winfo_children():
            widget.destroy()

        title = tk.Label(self.final_frame, text="Parámetros de Calibración por Cámara", font=("Arial", 16), bg="#f5f5f5")
        title.pack(pady=10)

        grid_frame = tk.Frame(self.final_frame, bg="#f5f5f5")
        grid_frame.pack(pady=10)

        row = 0
        col = 0
        for idx, (cam_name, params) in enumerate(camera_data.items()):
            card = tk.Frame(grid_frame, bg="white", bd=2, relief="groove", padx=10, pady=10)
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

            name_label = tk.Label(card, text=cam_name, font=("Arial", 12, "bold"), bg="white")
            name_label.pack(anchor="w")
            f = params.get("frame_id", "N/A")
            t = params["translation"]
            q = params["rotation"]
            frame_label = tk.Label(card, text=f"Frame ID: {f}", font=("Arial", 10), bg="white")
            frame_label.pack(anchor="w", pady=5)

            trans_label = tk.Label(card, text=f"Translación:\n  x: {t[0]:.3f}, y: {t[1]:.3f}, z: {t[2]:.3f}",
                                font=("Arial", 10), bg="white", justify="left")
            trans_label.pack(anchor="w", pady=5)

            rot_label = tk.Label(card, text=f"Rotación (quat):\n  w: {q[0]:.3f}, x: {q[1]:.3f}, y: {q[2]:.3f}, z: {q[3]:.3f}",
                                font=("Arial", 10), bg="white", justify="left")
            rot_label.pack(anchor="w")

            col += 1
            if col > 1:  # 2 columnas por fila
                col = 0
                row += 1

        # Botones
        button_frame = tk.Frame(self.final_frame, bg="#f5f5f5")
        button_frame.pack(pady=20)

        publish_button = tk.Button(button_frame, text="Publicar TF Estática", font=("Arial", 12),
                                bg="#4CAF50", fg="white", padx=10, pady=5, command=self.publish_tf)
        publish_button.grid(row=0, column=0, padx=10)

        exit_button = tk.Button(button_frame, text="Cancelar y Cerrar", font=("Arial", 12),
                                bg="#f44336", fg="white", padx=10, pady=5, command=self.root.quit)
        exit_button.grid(row=0, column=1, padx=10)

        restart_button = tk.Button(self.final_frame, text="Reiniciar Medición", font=("Arial", 12),
                                    bg="#FFC107", fg="black", padx=10, pady=5, command=self.restart)
        restart_button.pack(pady=10)

    def publish_tf(self):
        self.get_logger().info("Publicando transformaciones estáticas...")
        if not self.optimization_result:
            messagebox.showerror("Error", "No hay resultados de optimización disponibles.")
            return
        try:
            tf_broadcaster = BroadcastPoses.Request()
            tf_broadcaster.poses = self._parse_camera_transforms(self.optimization_result)
            future = self.tf_broadcaster_client.call_async(tf_broadcaster)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            if future.result().success:
                messagebox.showinfo("Éxito", f"Transformaciones estáticas publicadas correctamente.\n{future.result().message}")
            else:
                messagebox.showerror("Error", f"Error al publicar transformaciones: {future.result().message}")

        except Exception as e:
            self.get_logger().error(f"Error al publicar TF: {str(e)}")

    def restart(self):
        self.final_frame.pack_forget()
        self.main_frame.pack(fill='both', expand=True)
        self._update_result("Esperando nueva medición...")


    def metrics_callback(self, msg: String):
        self.get_logger().info(f"Recibido mensaje de métricas: {msg.data}")
        try:
            data = json.loads(msg.data)
            display = ""

            if isinstance(data, dict) and "cameras" in data:
                display += f"Error antes de optimizar: {data.get('error_before', 'N/A'):.4f}\n"
                display += f"Error después de optimizar: {data.get('error_after', 'N/A'):.4f}\n\n"
                display += "Poses de cámaras:\n"
                for cam_id, pose in data["cameras"].items():
                    t = pose["translation"]
                    frame_id = pose.get("frame_id", "N/A")
                    display += f"  - {cam_id}({frame_id}): ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})\n"
            else:
                display = msg.data

            self._update_result(display)
            self._update_final_result(display)

        except Exception as e:
            self._update_result("Error al interpretar el mensaje:\n" + str(e))

    def _update_result(self, text):
        self.result_box.config(state='normal')
        self.result_box.delete(1.0, tk.END)
        self.result_box.insert(tk.END, text)
        self.result_box.config(state='disabled')

    def _update_final_result(self, text):
        self.param_text.config(state='normal')
        self.param_text.delete(1.0, tk.END)
        self.param_text.insert(tk.END, text)
        self.param_text.config(state='disabled')

    def _parse_camera_transforms(self, json_data: dict) -> list:
        camera_transforms = []

        # Obtener el primer frame como el padre
        first_key = list(json_data.keys())[0]
        parent_frame = json_data[first_key]['frame_id']

        for cam_name, data in json_data.items():
            msg = CameraTransform()
            msg.parent_frame = parent_frame
            msg.child_frame = data['frame_id']  # Se asigna el frame actual como child_frame

            t = data["translation"]
            r = data["rotation"]

            msg.translation = Vector3(x=t[0], y=t[1], z=t[2])
            msg.rotation = Quaternion(x=r[1], y=r[2], z=r[3], w=r[0])  # ROS usa orden (x, y, z, w)

            camera_transforms.append(msg)

        return camera_transforms
    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    gui_node = CameraCalibratorGUI()
    gui_node.run()
    gui_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
