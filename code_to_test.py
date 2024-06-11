import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model

# Define the absolute path to the haarcascade file
cascade_path = r'C:\Users\Windownet\PycharmProjects\face_recognition\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'

# Load the Haar Cascade classifier
facedetect = cv2.CascadeClassifier(cascade_path)

# URL of the IP Webcam
url_ipwebcam = 'http://192.168.1.9:8080/shot.jpg'

# Load the pre-trained model
model = load_model('keras_model.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to get class name based on the prediction
def get_className(classNo):
    if classNo == 0:
        return "imane"
    elif classNo == 1:
        return "bono"

# Main loop
while True:
    # Fetch the frame from the IP webcam
    img_resp = urllib.request.urlopen(url_ipwebcam)
    img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    imgOrignal = cv2.imdecode(img_np, -1)

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
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the probability
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the result
    cv2.imshow("Result", imgOrignal)

    # Break the loop when 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the capture and destroy all windows
cv2.destroyAllWindows()
