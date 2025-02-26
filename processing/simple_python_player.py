import cv2

# Path to the video file
video_path = 'tennisvid.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Loop to read and display video frames
while True:
    ret, frame = cap.read()

    # Break the loop if no frames are returned (end of video)
    if not ret:
        break

    # Display the frame in a window
    cv2.imshow('Video Player', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()