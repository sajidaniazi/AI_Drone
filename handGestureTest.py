


import cv2
import mediapipe as mp
import time


# This sets up MediaPipe Hands, which finds hand landmarks.



mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
)

