import cv2
import mediapipe as mp
import time
from ultralytics import YOLO
from djitellopy import Tello

from drone_safety import (
    safe_move_up,
    safe_move_down,
    safe_move_forward,
    safe_hover,
    emergency_land
)

model = YOLO("yolov8n-cls.pt")
#bettery check

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

if tello.get_battery() < 20:
    print("Battery too low to fly")
    exit()

tello.takeoff()
tello.move_up(40) # change to 40

#tello camera connection
tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

last_command = "NONE"
last_time = 0
COOLDOWN = 1.5

def allow_command(command):
    global last_command, last_time
    current_time = time.time()
    if command != "NONE" and (command != last_command or (current_time - last_time > COOLDOWN)):
        last_command = command
        last_time = current_time
        return True
    return False

def get_finger_states(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = {}
    fingers["index"] = lm[8].y < lm[6].y
    fingers["middle"] = lm[12].y < lm[10].y
    fingers["ring"] = lm[16].y < lm[14].y
    fingers["pinky"] = lm[20].y < lm[18].y
    fingers["thumb_up"] = lm[4].y < lm[3].y
    return fingers

def detect_gesture(hand_landmarks):
    f = get_finger_states(hand_landmarks)
    index = f["index"]
    middle = f["middle"]
    ring = f["ring"]
    pinky = f["pinky"]
    thumb = f["thumb_up"]

    if index and middle and ring and pinky:
        return "MOVE_UP"
    elif index and not middle and not ring and not pinky:
        return "MOVE_DOWN"
    elif thumb and not index and not middle and not ring and not pinky:
        return "HOVER_AND_ANALYSE"
    elif not index and not middle and not ring and not pinky:
        return "MOVE_FORWARD_AND_SEARCH"

    return "NONE"
# telloc camera
while True:
    frame = frame_read.frame

    if frame is None:
        continue
#resize frame
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    command = "NONE"
#gesture detection
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            command = detect_gesture(hand_landmarks)

            if allow_command(command):
                print("COMMAND:", command)

                if command == "MOVE_UP":
                    safe_move_up(tello)

                elif command == "MOVE_DOWN":
                    safe_move_down(tello)

                elif command == "HOVER_AND_ANALYSE":
                    safe_hover()
                    print("Running YOLO analysis...")

                elif command == "MOVE_FORWARD_AND_SEARCH":
                    safe_move_forward(tello)

    cv2.putText(frame, f"Command: {command}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, "q =  quit", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)

    cv2.putText(frame, "x = land", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)

    cv2.imshow("Gesture Control - Tello Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        emergency_land(tello)
        break

    if key == ord('x'):
        print("Emergency Land")
        emergency_land(tello)
        break

tello.streamoff()
cv2.destroyAllWindows()