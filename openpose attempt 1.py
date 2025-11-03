import cv2
import sys
from openpose import pyopenpose as op

# OpenPose parameters
params = dict()
params["model_folder"] = "models/"  # Path to OpenPose models folder
params["hand"] = False
params["face"] = False

# Start OpenPose wrapper
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    
    # Display image with skeleton
    cv2.imshow("OpenPose Skeleton", datum.cvOutputData)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()