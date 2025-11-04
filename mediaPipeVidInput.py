import cv2
import mediapipe as mp
from pathlib import Path


INPUT_VIDEO  = r"C:\\Bono\\Ind Project\\Videos\\1000039767.mp4"             #input
OUTPUT_VIDEO = "output_skeleton.mp4"   
SHOW_WINDOW  = True       


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))
if not writer.isOpened():
    raise RuntimeError(f"Could not open output video for write: {OUTPUT_VIDEO}")

if SHOW_WINDOW:
    cv2.namedWindow("Pose Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Tracking", min(1280, w), min(720, h))


with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,          # 0=fastest, 2=most accurate
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)

        
        frame_out = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_out,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        # Write to output
        writer.write(frame_out)

        # Optional live preview
        if SHOW_WINDOW:
            cv2.imshow("Pose Tracking", frame_out)
            # Press 'q' to stop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Done! Saved to {Path(OUTPUT_VIDEO).resolve()}")