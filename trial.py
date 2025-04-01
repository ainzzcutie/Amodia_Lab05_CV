import cv2
import numpy as np
import time
import random

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the new iris image with an alpha channel
iris_images = [cv2.imread(f'eyes{i}.png', cv2.IMREAD_UNCHANGED) for i in range(1, 5)]
iris_colors = ["Blue", "Purple", "Red", "Yellow"]
current_iris_index = 0
current_iris = iris_images[current_iris_index]

# Initialize game variables
score = 0
start_time = time.time()
blink_detected = False
random_text_color = random.choice(iris_colors)

# Function to detect iris
def detect_iris(eye_roi):
    gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        iris_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(iris_contour)
    return None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) == 0 and not blink_detected:
            # Change iris color when eyes are closed
            current_iris_index = random.randint(0, 3)
            current_iris = iris_images[current_iris_index]
            blink_detected = True
        elif len(eyes) >= 2:
            blink_detected = False
        
        for (ex, ey, ew, eh) in eyes:
            eye_x, eye_y = x + ex, y + ey
            eye_roi = frame[eye_y:eye_y+eh, eye_x:eye_x+ew]
            
            iris = detect_iris(eye_roi)
            if iris:
                ix, iy, iw, ih = iris
                resized_iris = cv2.resize(current_iris, (iw, ih))
                iris_x, iris_y = eye_x + ix, eye_y + iy
                
                if resized_iris.shape[2] == 4:
                    alpha_s = resized_iris[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    
                    roi = frame[iris_y:iris_y+ih, iris_x:iris_x+iw]
                    for c in range(3):
                        roi[:, :, c] = (alpha_s * resized_iris[:, :, c] + alpha_l * roi[:, :, c])
    
    # Display the challenge text
    cv2.putText(frame, f"Match: {random_text_color}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Score: {score}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Check if the iris color matches the challenge text
    if iris_colors[current_iris_index] == random_text_color:
        if time.time() - start_time >= 5:
            score += 1
            random_text_color = random.choice(iris_colors)
            start_time = time.time()
    else:
        start_time = time.time()
    
    cv2.imshow('Blink Challenge Game', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
