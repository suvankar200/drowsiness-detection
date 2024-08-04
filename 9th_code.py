import cv2
import dlib
from scipy.spatial import distance
import serial
import time

# Load the pre-trained model for detecting faces
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate the ratio of eye aspect (EAR)
def calculate_ear(eye):
    # Calculate the distances between the vertical eye landmarks
    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])

    # Calculate the distance between the horizontal eye landmarks
    horizontal = distance.euclidean(eye[0], eye[3])

    # Calculate the eye aspect ratio
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


start_time = None
eyes_closed_duration = 0
status = "Open"

while True:
    # Get the current frame from the webcam
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert the frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    detected_faces = face_detector(grayscale_frame, 0)

    # Reset the timer if no face is detected
    if not detected_faces:
        start_time = None
        eyes_closed_duration = 0
        status = "Open"

    # Go through each detected face
    for face in detected_faces:
        # Predict the landmarks for the face
        landmarks = landmark_predictor(grayscale_frame, face)
        landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

        # Get the regions for each eye
        left_eye_landmarks = landmarks[36:42]
        right_eye_landmarks = landmarks[42:48]

        # Calculate the EAR for each eye
        left_eye_ear = calculate_ear(left_eye_landmarks)
        right_eye_ear = calculate_ear(right_eye_landmarks)

        # Calculate the average EAR for both eyes
        average_ear = (left_eye_ear + right_eye_ear) / 2.0

        # Set the thresholds for detecting if the eye is open or closed
        ear_threshold = 0.2

        # Check if the eyes are open or closed based on the EAR
        if average_ear < ear_threshold:
            if start_time is None:
                start_time = time.time()  # Start the timer
            else:
                eyes_closed_duration = time.time() - start_time
                if eyes_closed_duration >= 0:  # If eyes closed for 3 seconds
                    # Send command to Arduino to set digital pin 5 high
                    arduino.write(b'H')  # 'H' indicates eyes closed
                    status = "Closed"
        else:
            start_time = None  # Reset the timer if eyes open
            # Send command to Arduino to set digital pin 5 low
            arduino.write(b'L')  # 'L' indicates eyes open
            status = "Open"

        # Display status on the frame
        cv2.putText(frame, f"Eyes: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if status == "Open" else (0, 0, 255), 2)

        # Draw the landmarks on the frame
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Show the frame
    cv2.imshow("Eye Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
