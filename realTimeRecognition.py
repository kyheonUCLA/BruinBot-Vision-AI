from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

import google.cloud
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
import os
import sys
from urllib.request import urlopen

class RealTimeRecognition:
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

    def __init__(self) -> None:
        (self.face_detection, self.leg_detection, self.emotion_classifier) = self.load_models()

    def load_models(self):
        face_detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
        leg_detection_model_path = 'haarcascades/haarcascade_lowerbody.xml'
        emotion_model_path = 'models/emotion_model.hdf5'
        face_detection = cv2.CascadeClassifier(face_detection_model_path)
        leg_detection = cv2.CascadeClassifier(leg_detection_model_path)
        emotion_classifier = load_model(emotion_model_path, compile=False)
        return (face_detection, leg_detection, emotion_classifier)

    def process_frame(self, gray_frame, features):
            if len(features) > 0:
                features = sorted(features, reverse=True, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))[0]
                (fX, fY, fW, fH) = features
                roi = gray_frame[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                return roi, fX, fY, fW, fH
            else:
                raise Exception

    def run_in_local_mode(self):
        cv2.namedWindow("Local Main Window")
        self.camera = cv2.VideoCapture(0)

        if type(self.camera) == type(None):
            print('Invalid camera path was supplied')

        while True:
            frame = self.camera.read()[1]
            frame = imutils.resize(frame, width=400)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detection.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            try:
                roi, fX, fY, fW, fH = self.process_frame(gray_frame, faces)
                predictions = self.emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(predictions)
                emotion = RealTimeRecognition.EMOTIONS[predictions.argmax()]
                cv2.putText(frame, emotion, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
                widthStr = str(fW)
                heightStr = str(fH)
                print("faces," + emotion + "," + widthStr + "," + heightStr)
            except:
                cv2.imshow("Local Main Window", frame)

            cv2.imshow('Local Main Window', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def run_in_cloud_mode(self, credentials_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path # this line might be unnecessary
        cred = credentials.Certificate(credentials_path)
        app = initialize_app(cred)
        db = firestore.client() # Client or client? 'fair-myth-274206'

        cv2.namedWindow("Local Main Window")
        self.camera = cv2.VideoCapture(0)

        if type(self.camera) == type(None):
            print('Invalid camera path was supplied')

        while True:
            frame = self.camera.read()[1]
            frame = imutils.resize(frame, width=400)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detection.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            try:
                roi, fX, fY, fW, fH = self.process_frame(gray_frame, faces)
                predictions = self.emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(predictions)
                emotion = RealTimeRecognition.EMOTIONS[predictions.argmax()]
                cv2.putText(frame, emotion, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
                widthStr = str(fW)
                heightStr = str(fH)
                print("faces," + emotion + "," + widthStr + "," + heightStr)
                fb_data = { 
                    'timestamp' : firestore.SERVER_TIMESTAMP,
                    'emotion': emotion,
                    'width': widthStr,
                    'height': heightStr}
                data = db.collection('Faces')
                doc = data.document()
                doc.set(fb_data, merge=True)
            except:
                cv2.imshow("Local Main Window", frame)

            cv2.imshow('Local Main Window', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    
            
if __name__ == '__main__':
    obj = RealTimeRecognition()
    #obj.run_in_local_mode()
    obj.run_in_cloud_mode('editorKey.json')
