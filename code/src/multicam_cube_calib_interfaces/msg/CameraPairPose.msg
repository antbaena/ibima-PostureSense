# CameraPairPose.msg

# Timestamp y frame de referencia. 
# header.frame_id = id de la cámara "from" (ci)
std_msgs/Header header

# Identificadores de cámaras
string camera_from_id   # ci
string camera_to_id     # cj

# Transformación medida y su covarianza 6x6 (row-major).
# measured_transform.pose  = T_{ci -> cj}
# measured_transform.covariance = 36 elementos (m^2 y rad^2), orden [x, y, z, roll, pitch, yaw]
geometry_msgs/PoseWithCovariance measured_transform
