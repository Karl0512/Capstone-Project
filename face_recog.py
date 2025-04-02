import cv2
import numpy as np
import os
import json
import face_recognition
import time

# Load stored encodings
ENCODINGS_FILE = "face_encodings/encodings.npy"
NAMES_FILE = "face_encodings/names.json"

known_encodings = np.load(ENCODINGS_FILE, allow_pickle=True).tolist()
with open(NAMES_FILE, "r") as f:
    known_names = json.load(f)

def run_face_recognition():
    # Start webcam for face recognition
    video_capture = cv2.VideoCapture(0)

    print("Press 'q' to quit.")

    # Tracking recognized students
    recognized_students = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare with known encodings
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            color = (0, 0, 255)  # Default: Red for unknown

            # Find best match
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]  # Get student name
                if name not in recognized_students:  # Ensure one-time recognition
                    recognized_students.add(name)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"✅ {name} recognized at {timestamp}")
                    color = (0, 255, 0)  # Green for recognized faces

            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show frame
        cv2.imshow("Face Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_face_recognition()
