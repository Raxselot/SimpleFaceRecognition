import cv2
import face_recognition
import json
import numpy as np
import os

class FaceRecognitionManager:
    def __init__(self, storage_file="saved_faces.json"):
        self.saved_face_encodings = []
        self.saved_names = []
        self.storage_file = storage_file
        self.load_saved_faces()

    def add_person(self, image_path, person_name):
        if not os.path.exists(image_path):
            print(f"File {image_path} does not exist.")
            return

        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings: 
                self.saved_face_encodings.append(face_encodings[0])
                self.saved_names.append(person_name)
                self.save_faces()
            else:
                print(f"No face found in {image_path}.")
        except Exception as e:
            print(f"Error adding person: {e}")

    def load_saved_faces(self):
        if os.path.exists(self.storage_file):
            if os.path.getsize(self.storage_file) == 0: 
                print("The saved faces file is empty.")
            else:
                try:
                    with open(self.storage_file, 'r') as file:
                        data = json.load(file)
                        self.saved_face_encodings = [np.array(encoding) for encoding in data['encodings']]
                        self.saved_names = data['names']
                except json.JSONDecodeError as e:
                    print(f"Error reading the saved faces JSON file: {e}")
        else:
            print("No saved faces in the database. Starting with an empty database.")

    def save_faces(self):
        face_encodings_as_lists = [encoding.tolist() for encoding in self.saved_face_encodings]
        data = {'encodings': face_encodings_as_lists, 'names': self.saved_names}
        with open(self.storage_file, 'w') as file:
            json.dump(data, file)
            print("save faces written")
class VideoFaceRecognition:
    def __init__(self, face_manager):
        self.face_manager = face_manager
        self.camera = cv2.VideoCapture(0)

    def detect_faces(self):
        if not self.camera.isOpened():
            print("Camera couldn't be opened.")
            return

        while True:
            ret, frame = self.camera.read()

            if not ret:
                print("Error reading video stream.")
                break

            rgb_frame = frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_frame)
            print(f"Face locations: {face_locations}")  


            if not face_locations:
                print("No faces detected.")
                continue

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            print(f"Face encodings: {face_encodings}") 


            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.face_manager.saved_face_encodings, face_encoding)
                person_name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    person_name = self.face_manager.saved_names[first_match_index]

                self._draw_rectangle_with_name(frame, top, right, bottom, left, person_name)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def _draw_rectangle_with_name(self, frame, top, right, bottom, left, person_name):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, person_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Loading saved faces...")
    face_manager = FaceRecognitionManager()

    image_path = "/home/name/Scripts/Cheatsheets/test.jpg"  
    person_name = "Rosi"  
    face_manager.add_person(image_path, person_name)  

    print("Starting camera for face detection...")
    video_face_recognition = VideoFaceRecognition(face_manager)
    video_face_recognition.detect_faces()
