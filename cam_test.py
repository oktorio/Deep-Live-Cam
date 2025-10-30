import cv2

# 0 = default camera; try 1 or 2 if you have multiple cameras
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert BGR (OpenCV default) → RGB just to check colors
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow("OpenCV Camera Test", frame_rgb)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
