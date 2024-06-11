import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf

from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import speech_recognition as sr

# Define the absolute path to the haarcascade file
cascade_path = r'C:\Users\Windownet\PycharmProjects\face_recognition\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'

# Load the Haar Cascade classifier
facedetect = cv2.CascadeClassifier(cascade_path)

# Function to list available camera indices
def list_available_cameras():
    index = 0
    available_indices = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            available_indices.append(index)
        cap.release()
        index += 1
    return available_indices

# List available camera indices
available_camera_indices = list_available_cameras()
print("Available camera indices:", available_camera_indices)

# Use the first available camera index
if available_camera_indices:
    epoccam_index = available_camera_indices[0]  # Use the first available index
else:
    print("No camera detected.")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(epoccam_index)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height
font = cv2.FONT_HERSHEY_COMPLEX

# Load the pre-trained model
model = load_model('keras_model.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to get class name based on the prediction
def get_className(classNo):
    if classNo == 0:
        return "imane"
    elif classNo == 1:
        return "bono"

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Main loop
while True:
    success, imgOrignal = cap.read()
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = imgOrignal[y:y+h, x:x+w]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)

        # Make a prediction
        prediction = model.predict(img)
        classIndex = np.argmax(prediction)  # Use np.argmax to get the class index
        probabilityValue = np.amax(prediction)

        if classIndex in [0, 1]:
            detected_class = get_className(classIndex)
            print("Detected class:", detected_class)
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(detected_class), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the probability
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the result
    cv2.imshow("Result", imgOrignal)

    # Recognize speech from the microphone
    with sr.Microphone() as source:
        print("Listening...")
        audio_data = recognizer.listen(source)

    try:
        # Specify the language parameter for recognize_google
        text = recognizer.recognize_google(audio_data, language="fr-FR")
        print("You said:", text)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    # Break the loop when 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
