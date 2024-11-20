import cv2
import mediapipe as mp
import math
import keyboard
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

gesture_threshold = 30
time_between_gestures = 2

last_gesture_time = time.time()
gesture_active = False

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    height, width, _ = image.shape
    gesture_pinch = False
    gesture_pinky = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

            finger_landmarks = hand_landmarks.landmark
            index_tip = finger_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = finger_landmarks[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = finger_landmarks[mp_hands.HandLandmark.PINKY_TIP]

            ix, iy = int(index_tip.x * width), int(index_tip.y * height)
            px, py = int(pinky_tip.x * width), int(pinky_tip.y * height)
            tx, ty = int(thumb_tip.x * width), int(thumb_tip.y * height)

            distance_index_thumb = math.sqrt((tx - ix) ** 2 + (ty - iy) ** 2)
            distance_pinky = math.sqrt((px - ix) ** 2 + (py - iy) ** 2)

            if distance_index_thumb < gesture_threshold:
                gesture_pinch = True
            if distance_pinky < gesture_threshold:
                gesture_pinky = True

        current_time = time.time()

        if gesture_pinch and current_time - last_gesture_time >= time_between_gestures:
            if not gesture_active:
                keyboard.press('windows')
                keyboard.press('ctrl')
                gesture_active = True
            time.sleep(2)
            keyboard.press_and_release('O')

        elif gesture_active:
            keyboard.release('ctrl')
            gesture_active = False

        if gesture_pinky:
            keyboard.press_and_release('right')
            time.sleep(1)  # Ajusta el tiempo de espera entre acciones

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
