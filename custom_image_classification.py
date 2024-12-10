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
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define padding percentage
padding_ratio = 0.2

# Path to the custom image
image_path = 'input/images/khalid1.jpg'  
output_path = 'output/images/khalid1_ouput.jpg'  

# Load the image
image = cv2.imread(image_path)

if image is not None:
    # Convert to RGB for visualization and processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for x, y, w, h in faces:
            # Calculate padding
            pad_x = int(w * padding_ratio)
            pad_y = int(h * padding_ratio)

            # Expand the bounding box with padding
            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + w + pad_x, image.shape[1])
            y2 = min(y + h + pad_y, image.shape[0])

            # Crop the padded face region
            face_image = image_rgb[y1:y2, x1:x2]

            # Resize the face image to match the model input
            face_image_resized = cv2.resize(face_image, (256, 256))

            # Normalize and reshape the image
            face_image_array = np.expand_dims(face_image_resized / 255.0, axis=0)

            # Get predictions from the model
            predictions = model.predict(face_image_array)

            # Get the class with the highest prediction
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

            # Draw bounding box and prediction on the original image
            color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)  # Green for high confidence, Red otherwise
            label = f"{class_names[predicted_class]} ({confidence:.2f})" if confidence > 0.7 else "No Prediction"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert image back to BGR for saving
        output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Save the image with bounding boxes
        cv2.imwrite(output_path, output_image)

        print(f"Output image saved at {output_path}")
    else:
        print("No faces detected in the image.")
else:
    print(f"Could not load image at {image_path}. Please check the path.")
