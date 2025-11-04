from pathlib import Path
import cv2
import mediapipe as mp


input_dir = Path(r"C:\\Bono\\Ind Project\\Images")
output_dir = Path(r"C:\Bono\\Ind Project\\OutputImages")
output_dir.mkdir(parents=True, exist_ok=True)  

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

JOINTS_SPEC = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=6)  # BGR
BONES_SPEC  = mp_drawing.DrawingSpec(color=(255, 0, 0),   thickness=4)    

patterns = ("*.jpg", "*.jpeg", "*.png")

count = 0
with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                  min_detection_confidence=0.5) as pose:
    for pat in patterns:
        for img_path in input_dir.glob(pat):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Skipping unreadable file: {img_path}")
                continue

            # BGR -> RGB
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Draw landmarks if found
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=JOINTS_SPEC,
                    connection_drawing_spec=BONES_SPEC 
                )

            # Save to output folder with same filename
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), img)
            count += 1

print(f"Batch complete, Wrote {count} file(s) to: {output_dir}")