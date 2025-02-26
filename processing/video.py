import cv2

for i in range(5):  # Test indices 0-4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
        break
    else:
        print(f"No camera at index {i}")