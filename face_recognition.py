import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model  # type: ignore

# Load the trained face recognition model
model = load_model('face_recognition_model.keras')

# Load class labels from the pickle file
with open('class_labels.pkl', 'rb') as f:
    class_names = pickle.load(f)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define padding percentage
padding_ratio = 0.2

# Initialize webcam
webcam = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = webcam.read()

    # Detect faces in the original color frame
    faces = face_cascade.detectMultiScale(frame, minNeighbors=10, minSize=(100, 100))

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Calculate padding
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        # Expand the bounding box
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, frame.shape[1])
        y2 = min(y + h + pad_y, frame.shape[0])

        # Extract the face
        face_img = frame[y1:y2, x1:x2]

        # Preprocess the face image
        face_img_resized = cv2.resize(face_img, (256, 256))
        face_img_array = np.expand_dims(face_img_resized / 255.0, axis=0)

        # Get predictions
        predictions = model.predict(face_img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        if confidence > 0.8:
            class_name = class_names[predicted_class]
            color = (0, 255, 0)  # Green
        else:
            class_name = 'No Prediction'
            color = (0, 0, 255)  # Red

        # Draw the bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{class_name} ({confidence:.2f})', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the video frame
    cv2.imshow('Webcam Face Recognition', frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
