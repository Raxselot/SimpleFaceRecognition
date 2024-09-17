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
        self.load_faces()

    def add_person(self, face_encoding, person_name):
        self.saved_face_encodings.append(face_encoding)
        self.saved_names.append(person_name)
        self.save_faces()

    def load_faces(self):
        if not os.path.exists(self.storage_file) or os.path.getsize(self.storage_file) == 0:
            print("No saved faces found. Starting with an empty database.")
            return

        try:
            with open(self.storage_file, 'r') as file:
                data = json.load(file)
                self.saved_face_encodings = [np.array(encoding) for encoding in data['encodings']]
                self.saved_names = data['names']
        except json.JSONDecodeError as e:
            print(f"Error reading the saved faces JSON file: {e}")

    def save_faces(self):
        data = {
            'encodings': [encoding.tolist() for encoding in self.saved_face_encodings],
            'names': self.saved_names
        }
        with open(self.storage_file, 'w') as file:
            json.dump(data, file)
        print("Saved faces updated.")

class VideoFaceRecognition:
    def __init__(self, face_manager):
        self.face_manager = face_manager
        self.camera = self.start_camera()

    def start_camera(self):
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open video device.")
            return None
        print("Camera opened successfully.")
        return camera

    def detect_faces(self):
        if self.camera is None:
            return

        while True:
            frame = self.capture_frame()
            if frame is None:
                continue

            rgb_frame = self.frame_to_rgb(frame)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame)

            self.detected_faces(frame, face_locations, face_encodings)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def capture_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            print("Error reading video stream.")
            return None
        return frame

    def frame_to_rgb(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def detected_faces(self, frame, face_locations, face_encodings):
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            person_name = self.spot_person(face_encoding)
            self.draw_label(frame, top, right, bottom, left, person_name)

            if person_name == "Unknown":
                self.deal_with_unknown_person(face_encoding)

    def spot_person(self, face_encoding):
        matches = face_recognition.compare_faces(self.face_manager.saved_face_encodings, face_encoding)
        if True in matches:
            match_index = matches.index(True)
            return self.face_manager.saved_names[match_index]
        return "Unknown"

    def deal_with_unknown_person(self, face_encoding):
        user_input = input("Unbekanntes Gesicht erkannt. Möchtest du es speichern? (y/n): ")
        if user_input.lower() == 'y':
            person_name = input("Bitte gib den Namen der Person ein: ")
            self.face_manager.add_person(face_encoding, person_name)
            print(f"{person_name} wurde gespeichert.")

    def draw_label(self, frame, top, right, bottom, left, person_name):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, person_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_manager = FaceRecognitionManager()

    video_face_recognition = VideoFaceRecognition(face_manager)
    video_face_recognition.detect_faces()
