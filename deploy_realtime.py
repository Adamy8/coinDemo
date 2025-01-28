from ultralytics import YOLO
import cv2

# Load YOLO model
model_path = "best.pt"
model = YOLO(model_path)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define coin values (adjust based on your model's labels)
coin_values = {
    "penny": 0.01,
    "nickel": 0.05,
    "dime": 0.10,
    "quarter": 0.25
}

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Perform inference
    results = model(frame)
    frame_with_boxes = results[0].plot()  # Adds bounding boxes to the frame

    # Calculate total coin value
    total_value = 0.0
    for result in results[0].boxes:
        class_id = int(result.cls)  # Class ID of the detected object
        label = model.names[class_id]  # Get the label (e.g., "penny", "nickel")
        if label in coin_values:  # Ensure the label corresponds to a known coin
            total_value += coin_values[label]

    # Format the total value as a string
    total_value_str = f"You have: ${total_value:.2f}!"

    # Overlay the total value on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (30, frame.shape[0] - 40)  # Bottom-left corner
    font_scale = 2
    font_color = (0, 255, 0)  # Green text
    line_type = 3

    cv2.putText(frame_with_boxes, total_value_str, position, font, font_scale, font_color, line_type)

    # Display the frame with bounding boxes and total value
    cv2.imshow("Detection", frame_with_boxes)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()
