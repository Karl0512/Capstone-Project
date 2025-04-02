import cv2
import numpy as np
import os
import json
import face_recognition

# Folder to store face encodings
ENCODINGS_FOLDER = "face_encodings"
ENCODINGS_FILE = os.path.join(ENCODINGS_FOLDER, "encodings.npy")
NAMES_FILE = os.path.join(ENCODINGS_FOLDER, "names.json")

# Ensure encoding folder exists
os.makedirs(ENCODINGS_FOLDER, exist_ok=True)

# Load existing encodings if available
if os.path.exists(ENCODINGS_FILE) and os.path.exists(NAMES_FILE):
    known_encodings = np.load(ENCODINGS_FILE, allow_pickle=True).tolist()
    with open(NAMES_FILE, "r") as f:
        known_names = json.load(f)
else:
    known_encodings = []
    known_names = []

def enroll_face():
    # Start webcam for face enrollment
    video_capture = cv2.VideoCapture(0)
    print("Press 's' to save face. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using face_recognition
        face_locations = face_recognition.face_locations(rgb_frame)

        for top, right, bottom, left in face_locations:
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Face Enrollment", frame)

        # Press 's' to save face
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and face_locations:
            for top, right, bottom, left in face_locations:
                face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])

                if face_encoding:
                    name = input("Enter student name: ")  # Get student name
                    known_encodings.append(face_encoding[0])  # Append encoding
                    known_names.append(name)  # Append name

                    # Save to file
                    np.save(ENCODINGS_FILE, np.array(known_encodings, dtype=object))
                    with open(NAMES_FILE, "w") as f:
                        json.dump(known_names, f)

                    print(f"✅ Saved {name}'s face!")

        # Press 'q' to exit
        if key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    enroll_face()
