import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializar el modelo de MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Configuración de la cámara
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Definir el tamaño del marco
frame_width, frame_height = 640, 480

# Definir el tamaño del rectángulo delimitador
bounding_box_width, bounding_box_height = 300, 200

# Calcular las coordenadas del rectángulo delimitador en el centro del marco
bounding_box = {
    'left': int((frame_width - bounding_box_width) / 2),
    'top': int((frame_height - bounding_box_height) / 2),
    'right': int((frame_width + bounding_box_width) / 2),
    'bottom': int((frame_height + bounding_box_height) / 2)
}

# Número de muestras para el filtro de media móvil
num_samples = 5
# Listas para almacenar las coordenadas anteriores
prev_x_samples = [0] * num_samples
prev_y_samples = [0] * num_samples
alpha = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Convertir la imagen a RGB para MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Detectar manos en la imagen
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Dimensiones del frame
    height, width, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los landmarks de la mano
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

            # Obtener las coordenadas de los landmarks de los dedos
            finger_landmarks = hand_landmarks.landmark
            index_tip = finger_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = finger_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = finger_landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]

            # Coordenadas de los dedos para realizar acciones
            ix, iy = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])
            mx, my = int(middle_tip.x * image.shape[1]), int(middle_tip.y * image.shape[0])
            rx, ry = int(ring_tip.x * image.shape[1]), int(ring_tip.y * image.shape[0])

            # Verificar si el dedo índice está dentro del rectángulo delimitador
            if bounding_box['left']+5 < ix < bounding_box['right']+5 and bounding_box[
                'top'] < iy < \
                    bounding_box['bottom']:
                # Escalar las coordenadas al tamaño de la pantalla
                screen_x = int(
                    (ix - bounding_box['left']) / (bounding_box_width) * pyautogui.size().width)
                screen_y = int(
                    (iy - bounding_box['top']) / (bounding_box_height) * pyautogui.size().height)

                # Aplicar un filtro de media móvil a las coordenadas del puntero
                prev_x_samples.pop(0)
                prev_y_samples.pop(0)
                prev_x_samples.append(screen_x)
                prev_y_samples.append(screen_y)
                smoothed_x = int(sum(prev_x_samples) / num_samples)
                smoothed_y = int(sum(prev_y_samples) / num_samples)

                # Dentro del bucle while donde se actualizan las coordenadas del puntero
                smoothed_x = alpha * smoothed_x + (1 - alpha) * screen_x
                smoothed_y = alpha * smoothed_y + (1 - alpha) * screen_y

                # Invertir horizontalmente las coordenadas del puntero
                inverted_x = pyautogui.size().width - smoothed_x
                # Dibujar un círculo si el índice y medio están juntos
                distance_threshold = 30
                if abs(ix - mx) < distance_threshold and abs(iy - my) < distance_threshold:
                    cv2.circle(image, (ix, iy), 10, (255, 0, 0), -1)
                    print("Activo puntero")
                    pyautogui.moveTo(smoothed_x, smoothed_y)
                    # Hacer clic si el anular se dobla
                    if ry < my:
                       cv2.putText(image, "Haciendo clic", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                       cv2.circle(image, (ix, iy), 12, (255, 255, 255),2)  # Cambia el valor de 2 según el grosor del borde que desees

    # Dibujar el rectángulo delimitador en el marco
    cv2.rectangle(image, (bounding_box['left'], bounding_box['top']), (bounding_box['right'], bounding_box['bottom']),
                  (0, 255, 0), 2)
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
