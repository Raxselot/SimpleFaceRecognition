import cv2
import face_recognition
import numpy as np
import os
import pickle
from typing import List, Optional, Tuple

class FaceRecognitionManager:
    def __init__(self, storage_file: str = "saved_faces.pkl"):

        self.storage_file = storage_file
        self.saved_face_encodings: List[np.ndarray] = []
        self.saved_names: List[str] = []
        self.load_faces()

    def add_person(self, face_encoding: np.ndarray, person_name: str) -> None:

        self.saved_face_encodings.append(face_encoding)
        self.saved_names.append(person_name)
        self.save_faces()

    def load_faces(self) -> None:

        if not os.path.exists(self.storage_file) or os.path.getsize(self.storage_file) == 0:
            print("Keine gespeicherten Gesichter gefunden. Starte mit einer leeren Datenbank.")
            return

        try:
            with open(self.storage_file, 'rb') as file:
                data = pickle.load(file)
                self.saved_face_encodings = data['encodings']
                self.saved_names = data['names']
            print(f"{len(self.saved_names)} Gesichter erfolgreich geladen.")
        except (pickle.UnpicklingError, KeyError) as e:
            print(f"Fehler beim Lesen der gespeicherten Gesichter: {e}")

    def save_faces(self) -> None:

        data = {
            'encodings': self.saved_face_encodings,
            'names': self.saved_names
        }
        with open(self.storage_file, 'wb') as file:
            pickle.dump(data, file)
        print("Gespeicherte Gesichter aktualisiert.")

class VideoFaceRecognition:
    def __init__(self, face_manager: FaceRecognitionManager, frame_resize_scale: float = 0.25):

        self.face_manager = face_manager
        self.camera = self.start_camera()
        self.frame_resize_scale = frame_resize_scale
        self.unknown_faces_encodings: List[np.ndarray] = []
        self.unknown_faces_names: List[str] = []

    def start_camera(self) -> Optional[cv2.VideoCapture]:

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Fehler: Konnte das Videogerät nicht öffnen.")
            return None
        print("Kamera erfolgreich geöffnet.")
        return camera

    def detect_faces(self) -> None:

        if self.camera is None:
            return

        try:
            while True:
                frame = self.capture_frame()
                if frame is None:
                    continue

                rgb_small_frame = self.frame_to_rgb_small(frame)
                face_locations, face_encodings = self.get_face_locations_encodings(rgb_small_frame)

                self.process_detected_faces(frame, face_locations, face_encodings)

                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cleanup()

    def capture_frame(self) -> Optional[np.ndarray]:

        ret, frame = self.camera.read()
        if not ret:
            print("Fehler beim Lesen des Videostroms.")
            return None
        return frame

    def frame_to_rgb_small(self, frame: np.ndarray) -> np.ndarray:

        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resize_scale, fy=self.frame_resize_scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        return rgb_small_frame

    def get_face_locations_encodings(self, rgb_frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return face_locations, face_encodings

    def process_detected_faces(self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]], face_encodings: List[np.ndarray]) -> None:

        for face_location, face_encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = [int(coord / self.frame_resize_scale) for coord in face_location]
            person_name = self.spot_person(face_encoding)

            self.draw_label(frame, top, right, bottom, left, person_name)

            if person_name == "Unbekannt":
                self.handle_unknown_person(face_encoding)

    def spot_person(self, face_encoding: np.ndarray) -> str:

        if not self.face_manager.saved_face_encodings:
            return "Unbekannt"

        matches = face_recognition.compare_faces(self.face_manager.saved_face_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(self.face_manager.saved_face_encodings, face_encoding)
        
        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                return self.face_manager.saved_names[best_match_index]
        return "Unbekannt"

    def handle_unknown_person(self, face_encoding: np.ndarray) -> None:
 
        
        self.unknown_faces_encodings.append(face_encoding)
        #TODO Building a Logik that saves Unknown Faces after some time 

    def draw_label(self, frame: np.ndarray, top: int, right: int, bottom: int, left: int, person_name: str) -> None:

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, person_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    def cleanup(self) -> None:

        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Ressourcen freigegeben und Fenster geschlossen.")

def main():

    face_manager = FaceRecognitionManager()
    video_face_recognition = VideoFaceRecognition(face_manager)
    video_face_recognition.detect_faces()



if __name__ == "__main__":
    main()
