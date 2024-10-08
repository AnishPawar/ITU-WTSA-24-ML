import cv2
import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from PIL import Image  # Importing PIL explicitly for better image handling

# Load the trained model (assuming MobileNetV2 as per your earlier code)
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 output classes (Male, Female)

# Load the saved model state dict
model.load_state_dict(torch.load('model_75_xx.pt', map_location=torch.device('cpu')))  # Ensure model loads properly
model.eval()  # Set model to evaluation mode

# Send model to the appropriate device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model

# Define the transformations (must match the training preprocessing pipeline)
preprocess = transforms.Compose([
    transforms.ToPILImage(),               # Convert OpenCV image to PIL format
    transforms.Resize((224, 224)),         # Resize to the input size
    transforms.ToTensor(),                 # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

class_names = ['Female', 'Male']

# Load the pre-trained Haarcascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the video stream
    ret, frame = cap.read()
    
    # Convert the frame to grayscale (Haarcascades work better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)


        image_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Preprocess the image (resize, normalize, etc.)
        input_tensor = preprocess(image_rgb)  # Apply transforms
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
        input_tensor = input_tensor  # Move to the appropriate device

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)

        # Get the predicted class
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        image = cv2.putText(frame, f'{predicted_class}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 
                   2, (0,0,255), 2, cv2.LINE_AA)
        # Display the prediction
        # print(f'Predicted Class: {predicted_class}')

    # Display the frame with the detected faces
    cv2.imshow('Face Detection', frame)
    
    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()