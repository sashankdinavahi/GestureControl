import cv2
import mediapipe as mp
import pyautogui
import time
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Gesture Definitions
THUMBS_UP_GESTURE = [1, 0, 0, 0, 1]
THUMBS_DOWN_GESTURE = [0, 0, 0, 0, 0]
CLOSED_FIST_GESTURE = [0, 0, 0, 0, 0]

# Tab Change Variables
TAB_CHANGE_THRESHOLD = 0.1
previous_x = None
TAB_CHANGE_DELAY = 0.5

# Zoom Variables
ZOOM_THRESHOLD = 0.05  # Adjust for pinch sensitivity
previous_distance = None
ZOOM_DELAY = 0.5

def detect_gesture(hand_landmarks):
    finger_states = []

    # Thumb
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    finger_states.append(1 if thumb_tip < thumb_ip else 0)  # Thumb extended if tip < IP

    # Index, Middle, Ring, Pinky (Specific states for each gesture)
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    finger_states.append(1 if index_tip < index_dip and index_tip < index_pip else 0)  # Index extended

    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    finger_states.append(1 if middle_tip < middle_dip and middle_tip < middle_pip else 0)  # Middle extended

    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    finger_states.append(1 if ring_tip < ring_dip and ring_tip < ring_pip else 0)  # Ring extended

    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    finger_states.append(1 if pinky_tip < pinky_dip and pinky_tip < pinky_pip else 0)  # Pinky extended

    if finger_states == THUMBS_UP_GESTURE:
        return "scroll_up"
    elif finger_states == THUMBS_DOWN_GESTURE:
        return "scroll_down"
    elif finger_states == CLOSED_FIST_GESTURE:
        return "stop"
    else:
        return "none"

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow("Hand Gesture Scroll")

scroll_amount = 3
scroll_delay = 0.1
last_tab_switch = 0
last_zoom_change = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = detect_gesture(hand_landmarks)

            index_finger_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            if gesture == "scroll_up":
                cv2.setWindowProperty("Hand Gesture Scroll", cv2.WND_PROP_TOPMOST, 1)
                time.sleep(scroll_delay)
                pyautogui.scroll(scroll_amount)
                cv2.putText(frame, "Scrolling Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gesture == "scroll_down":
                cv2.setWindowProperty("Hand Gesture Scroll", cv2.WND_PROP_TOPMOST, 1)
                time.sleep(scroll_delay)
                pyautogui.scroll(-scroll_amount)
                cv2.putText(frame, "Scrolling Down", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif gesture == "stop":
                cv2.putText(frame, "Scrolling Stopped", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif previous_x is not None:
                delta_x = index_finger_x - previous_x

                if abs(delta_x) > TAB_CHANGE_THRESHOLD and time.time() - last_tab_switch > TAB_CHANGE_DELAY:
                    if delta_x > 0:
                        pyautogui.hotkey('ctrl', 'shift', 'tab')
                        cv2.putText(frame, "Next Tab", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        pyautogui.hotkey('ctrl', 'tab')
                        cv2.putText(frame, "Previous Tab", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                    last_tab_switch = time.time()
            previous_x = index_finger_x

            # Zoom (Pinch Gesture)
            distance = calculate_distance(thumb_tip, index_finger_tip)

            if previous_distance is not None:
                delta_distance = distance - previous_distance

                if abs(delta_distance) > ZOOM_THRESHOLD and time.time() - last_zoom_change > ZOOM_DELAY:
                    if delta_distance > 0: # Pinch out (zoom in)
                        pyautogui.hotkey('ctrl', '+')  # or 'command', '+' on macOS
                        cv2.putText(frame, "Zoom In", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else: # Pinch in (zoom out)
                        pyautogui.hotkey('ctrl', '-')  # or 'command', '-' on macOS
                        cv2.putText(frame, "Zoom Out", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    last_zoom_change = time.time()

            previous_distance = distance

    cv2.imshow("Hand Gesture Scroll", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Correctly closed if statement
        break  # Break the loop

cap.release()
cv2.destroyAllWindows()
hands.close()
