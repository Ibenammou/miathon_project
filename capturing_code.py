import cv2
import os

# Define the absolute path to the haarcascade file
cascade_path = r'C:\Users\Windownet\PycharmProjects\face_recognition\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'

# Debug output to verify the path
print(f"Loading Haar Cascade from: {cascade_path}")

# Check if the file exists and is accessible
if not os.path.exists(cascade_path):
    print(f"Error: The file {cascade_path} does not exist.")
else:
    print(f"The file {cascade_path} exists and will be loaded.")

# Load the Haar Cascade
facedetect = cv2.CascadeClassifier(cascade_path)

# Check if the classifier loaded correctly
if facedetect.empty():
    print("Error: Failed to load the Haar Cascade classifier.")
    exit(1)  # Exit the script if the classifier could not be loaded
else:
    print("Haar Cascade classifier loaded successfully.")

video = cv2.VideoCapture(0)

count = 0

nameID = str(input("Enter Your Name: ")).lower()

path = 'images/' + nameID

isExist = os.path.exists(path)

if isExist:
    print("Name Already Taken")
    nameID = str(input("Enter Your Name Again: "))
    path = 'images/' + nameID
    os.makedirs(path, exist_ok=True)  # Ensure the directory exists
else:
    os.makedirs(path)

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to capture frame from camera.")
        break
    
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        count += 1
        name = os.path.join(path, f"{count}.jpg")
        print(f"Creating Image: {name}")
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    cv2.imshow("WindowFrame", frame)
    cv2.waitKey(1)
    
    if count > 500:
        break

video.release()
cv2.destroyAllWindows()
