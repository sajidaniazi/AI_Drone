import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class HandGestureDetector:

    def __init__(self, min_detection_confidence=0.75, min_tracking_confidence=0.75):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]

    def _finger_up(self, lm, tip_id, pip_id):
        """True if a finger tip is above its PIP knuckle (y is inverted in image space)."""
        return lm[tip_id].y < lm[pip_id].y

    def classify(self, landmarks):
        """
         'hand_up', 'hand_down', or None.
        landmarks
        """
        lm = landmarks.landmark
        fingers_extended = [
            self._finger_up(lm, tip, pip)
            for tip, pip in zip(self.FINGER_TIPS, self.FINGER_PIPS)
        ]
        num_up = sum(fingers_extended)

        if num_up >= 4:
            return "hand_up"
        elif num_up == 0:
            return "hand_down"
        return None  # transitioning

    def process_frame(self, frame):

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            gesture = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the 21-point skeleton
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    gesture = self.classify(hand_landmarks)

            return frame, gesture

        def release(self):
            self.hands.close()

    if __name__ == "__main__":
        detector = HandGestureDetector()
        cap = cv2.VideoCapture(0)

        print("Hand gesture test running and Q to quit.")
        print("  Open hand  → HAND UP  (drone rises)")
        print("  Fist       → HAND DOWN (drone descends)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            annotated, gesture = detector.process_frame(frame)

            # gesture label
            label = gesture if gesture else "---"
            colour = (0, 220, 80) if gesture == "hand_up" else \
                (0, 80, 220) if gesture == "hand_down" else \
                    (180, 180, 180)

            cv2.putText(annotated, label, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3)
            cv2.imshow("Hand Gesture Test", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        detector.release()
        cv2.destroyAllWindows()
