import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model  # type: ignore

# Load the trained face recognition model
model = load_model("face_recognition_model.keras")

# Load class labels from the pickle file
with open("class_labels.pkl", "rb") as f:
    class_names = pickle.load(f)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Define padding percentage
padding_ratio = 0.2

# Path to the video file
video_path = "input/videos/shafin1.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}.")
    exit()

# Output video setup
output_path = "output/videos/shafin1_output.mp4"  # Replace with desired output path
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (frame_width, frame_height),
)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        # Calculate padding
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        # Expand the bounding box with padding
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, frame.shape[1])
        y2 = min(y + h + pad_y, frame.shape[0])

        # Crop the face region
        face_image = frame[y1:y2, x1:x2]

        # Resize and preprocess the face image
        face_image_resized = cv2.resize(face_image, (256, 256))
        face_image_array = np.expand_dims(face_image_resized / 255.0, axis=0)

        # Get predictions
        predictions = model.predict(face_image_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Show bounding box and label only if confidence > 0.9
        if confidence > 0.9:
            class_name = class_names[predicted_class]
            color = (0, 255, 0)  # Green for high confidence
            label = f"{class_name} ({confidence:.2f})"

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Video Face Recognition", frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_path}.")
