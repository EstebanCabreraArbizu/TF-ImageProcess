import cv2 # OpenCV para procesamiento de imágenes y vídeo
import mediapipe as mp #para detección y seguimiento de manos
import pyautogui #para el control del cursor del mouse y acciones en pantalla
import math
import glob
import os

cap = cv2.VideoCapture(0) #para capturar imágenes desde la webcam.

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializar PyAutoGUI
pyautogui.FAILSAFE = False  # Desactivar la función de seguridad de PyAutoGUI

# Obtener el tamaño de la pantalla
screen_width, screen_height = pyautogui.size()

# Definir las coordenadas del rectángulo delimitador
rectangle_coordinates = [(0, 0), (screen_width, screen_height)]

# Ajustar la sensibilidad del movimiento del puntero
scale_factor = 2

# Definir umbrales para gestos
OPEN_HAND_THRESHOLD = 0.1  # Umbral para determinar si la mano está abierta o cerrada
FINGERS_TOGETHER_DISTANCE_THRESHOLD = 0.1  # Umbral para determinar si el índice y medio están juntos
GESTURE_THRESHOLD = 200

# Definir gestos
GESTURES = {
    "thumbs_up": "thumbs_up",
    "thumbs_down": "thumbs_down",
    "peace_sign": "peace_sign",
    "handshake": "handshake",
}

# Estado anterior de los dedos índice y medio
prev_fingers_together = False

react_folder = 'emojis'
react_files = glob.glob(os.path.join(react_folder, '*.png'))

def open_image(gesture): # muestra una imagen en pantalla basada en el gesto detectado
    if gesture in GESTURES.values():
        # Obtener el índice de la imagen correspondiente al gesto
        image_index = list(GESTURES.values()).index(gesture)
        image_path = react_files[image_index]

        # Mostrar la imagen en pantalla completa
        image = cv2.imread(image_path)
        cv2.imshow('Imagen', image)
        cv2.waitKey(4000)  # Esperar 1 segundo antes de cambiar de imagen

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el fotograma")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        # Calcular el centro del frame
        center_x = int(frame.shape[1] / 2)
        center_y = int(frame.shape[0] / 2)

        # Calcular la mitad de la anchura y altura del rectángulo
        half_width = int((rectangle_coordinates[1][0] - rectangle_coordinates[0][0]) / 2)
        half_height = int((rectangle_coordinates[1][1] - rectangle_coordinates[0][1]) / 2)

        # Calcular las nuevas coordenadas del rectángulo delimitador
        new_rectangle_coordinates = [
            (center_x - half_width, center_y - half_height),
            (center_x + half_width, center_y + half_height)
        ]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener las coordenadas de los extremos de los dedos
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP] #detecta el extremo del pulgar

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.line(frame, (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])),
                         (int(middle_finger_tip.x * frame.shape[1]), int(middle_finger_tip.y * frame.shape[0])), (0, 255, 0), 2)

                # Calcula la distancia entre los dedos índice y medio
                distance_clik_finger = math.dist((index_finger_tip.x * frame.shape[1], index_finger_tip.y * frame.shape[0]),
                                     (middle_finger_tip.x * frame.shape[1], middle_finger_tip.y * frame.shape[0]))

                # Obtener las coordenadas en píxeles
                index_finger_tip_pixel = (int(index_finger_tip.x * screen_width * scale_factor),
                                            int(index_finger_tip.y * screen_height * scale_factor))
                middle_finger_tip_pixel = (int(middle_finger_tip.x * screen_width * scale_factor),
                                            int(middle_finger_tip.y * screen_height * scale_factor))

                # Calcular el promedio de las coordenadas de los extremos de los dedos
                pointer_x = (index_finger_tip_pixel[0] + middle_finger_tip_pixel[0]) // 2
                pointer_y = (index_finger_tip_pixel[1] + middle_finger_tip_pixel[1]) // 2
                
                # print(f"Distancia entre dedos: ({distance_clik_finger})")
                # # Verifica si la mano está cerca de la región de los gestos
                # if distance_clik_finger < 40 and pointer_y > screen_height - GESTURE_THRESHOLD:
                #     pyautogui.click()
                #     print("se realizo clik")


                # Calcular la distancia entre los extremos de los dedos índice y medio
                distance_fingers_together = ((index_finger_tip.x - middle_finger_tip.x)**2 +
                                              (index_finger_tip.y - middle_finger_tip.y)**2)**0.5
                distance_thumb_index_y = thumb_tip.y - index_finger_tip.y
                distance_thumb_index_x = thumb_tip.x - index_finger_tip.x

                # Detectar si la mano está abierta o cerrada
                hand_open = distance_fingers_together > OPEN_HAND_THRESHOLD

                # Mover el puntero si la mano está abierta
                # if hand_open:
                    
                #     print(f"Coordenadas del puntero: ({pointer_x}, {pointer_y}")
                #     # Mover el puntero del ratón
                #     pyautogui.moveTo(pointer_x, pointer_y)

                # Identificar el gesto realizado
                if distance_thumb_index_y < 0.05:
                    if distance_thumb_index_x > 0.1:
                        gesture = GESTURES["handshake"]
                    else:
                        gesture = GESTURES["thumbs_up"]
                elif distance_thumb_index_y > 0.05:
                    gesture = GESTURES["thumbs_down"]
                elif index_finger_tip.x < middle_finger_tip.x:
                        gesture = GESTURES["peace_sign"]
                else:
                    gesture = None

                # Mostrar el gesto
                print(f"Gesto: {gesture}")
                open_image(gesture)

            # Dibujar el rectángulo delimitador en el frame
            cv2.rectangle(frame, new_rectangle_coordinates[0], new_rectangle_coordinates[1], (0, 255, 0), 2)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
