from pathlib import Path
import cv2
import mediapipe as mp

input_dir = Path("images")
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(static_image_mode=True) as pose:
    for img_path in input_dir.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None: continue

        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imwrite(str(output_dir / img_path.name), img)

print("Batch complete")