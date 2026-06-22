import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def finger_up(tip, pip, landmarks):
    return landmarks[tip].y < landmarks[pip].y


def is_hand_heart(hand_landmarks_list):

    if len(hand_landmarks_list) < 2:
        return False

    lm1 = hand_landmarks_list[0].landmark
    lm2 = hand_landmarks_list[1].landmark

    # Index tips close together (top of heart)
    dx = lm1[8].x - lm2[8].x
    dy = lm1[8].y - lm2[8].y
    idx_dist = (dx*dx + dy*dy)**0.5

    # Thumb tips close together (bottom of heart)
    dx = lm1[4].x - lm2[4].x
    dy = lm1[4].y - lm2[4].y
    thumb_dist = (dx*dx + dy*dy)**0.5

    # Middle, ring, pinky harus ditekuk (tidak naik)
    mid_up1 = finger_up(12, 10, lm1)
    mid_up2 = finger_up(12, 10, lm2)
    ring_up1 = finger_up(16, 14, lm1)
    ring_up2 = finger_up(16, 14, lm2)
    pinky_up1 = finger_up(20, 18, lm1)
    pinky_up2 = finger_up(20, 18, lm2)

    return (
        idx_dist < 0.1 and
        thumb_dist < 0.15 and
        not mid_up1 and not mid_up2 and
        not ring_up1 and not ring_up2 and
        not pinky_up1 and not pinky_up2
    )


cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    heart_detected = False

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) >= 2:

        if is_hand_heart(result.multi_hand_landmarks):
            heart_detected = True

    if heart_detected:

        frame = cv2.GaussianBlur(
            frame,
            (61, 61),
            0
        )

    cv2.imshow(
        "Heart Blur",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
