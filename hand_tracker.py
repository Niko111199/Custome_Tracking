import time
import cv2
import mediapipe as mp


CAM_ID = 0          
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Handtracker initialization
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF
)


FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
FINGER_PIPS = [3, 6, 10, 14, 18]  

def normalized_to_pixel(norm_x, norm_y, width, height):
    return int(norm_x * width), int(norm_y * height)

def count_fingers(hand_landmarks, img_w, img_h, handedness_label=None):

    landmarks = hand_landmarks.landmark
    fingers = []

    if handedness_label is not None and 'Left' in handedness_label:
        thumb_open = landmarks[FINGER_TIPS[0]].x > landmarks[FINGER_PIPS[0]].x
    else:
        thumb_open = landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_PIPS[0]].x
    fingers.append(1 if thumb_open else 0)

    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        tip_y = landmarks[tip_idx].y
        pip_y = landmarks[pip_idx].y
        fingers.append(1 if tip_y < pip_y else 0)

    return sum(fingers), fingers
cap = cv2.VideoCapture(CAM_ID)
prev_time = 0.0

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Cammera not found!")
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]


        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Count fingers
                label = handedness.classification[0].label  # 'Left' or 'Right'
                count, per_finger = count_fingers(hand_landmarks, img_w, img_h, handedness_label=label)

                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(xs) * img_w), int(max(xs) * img_w)
                y_min, y_max = int(min(ys) * img_h), int(max(ys) * img_h)

                # draw info
                cv2.rectangle(frame, (x_min-10, y_min-30), (x_min+140, y_min), (0,0,0), -1)
                cv2.putText(frame, f'{label} hand: {count} fingers', (x_min-5, y_min-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Specific gesture detection
                # Thumbs up 
                if per_finger == [1,0,0,0,0]:
                    cv2.putText(frame, 'Thumbs UP!', (x_min, y_max+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    
                # Peace sign
                if per_finger == [0,1,1,0,0]:
                    cv2.putText(frame, 'Peace Sign!', (x_min, y_max+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
                    
                # Fist
                if per_finger == [0,0,0,0,0]:
                    cv2.putText(frame, 'Fist!', (x_min, y_max+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    
                # meatal horns
                if per_finger == [0,1,0,0,1]:
                    cv2.putText(frame, 'Metal Horns!', (x_min, y_max+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
                    
                # question
                if per_finger == [0,1,0,0,0]:
                    cv2.putText(frame, 'Question?', (x_min, y_max+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2)
                
                #high five
                if per_finger == [1,1,1,1,1]:
                    cv2.putText(frame, 'High Five!', (x_min, y_max+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow('HandTracker', frame)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty('HandTracker', cv2.WND_PROP_VISIBLE) < 1 or key == 27: 
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
