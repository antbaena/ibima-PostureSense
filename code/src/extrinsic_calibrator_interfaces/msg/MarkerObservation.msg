# MarkerObservation.msg

# Cabecera con timestamp y frame_id (de la cámara)
std_msgs/Header header

# ID del marcador detectado
int32 marker_id

# Nombre o ID de la cámara
string camera_id

# Transformación del marcador respecto a la cámara
geometry_msgs/Transform transform

# Confianza en la observación [0.0 - 1.0]
float32 confidence

# Distancia aproximada en metros (desde la cámara)
float32 distance

# Tamaño proyectado en píxeles cuadrados
float32 pixel_area
