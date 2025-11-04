import cv2
import mediapipe as mp 

# Mediapipe pose setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Start webcam feed
cap = cv2.VideoCapture(0)


window_width = 1280
window_height = 720

cv2.namedWindow("Pose Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Tracking", window_width, window_height)


with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        #convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

        cv2.imshow('Pose Tracking', image)

        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
