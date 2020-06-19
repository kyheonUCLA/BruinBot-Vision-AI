from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

import google.cloud
import firebase_admin
from firebase_admin import credentials, firestore
import os
import sys
from urllib2 import urlopen


class RealTimeRecognition:
    def __init__(self, camera_url):
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./key2.json" # ENTER PATH LOCATION IF RUN LOCALLY
        cred = credentials.Certificate("./key.json") #ENTER PATH LOCATION TO CREDENTIALS HERE
        app = firebase_admin.initialize_app(cred)

        self.store = firestore.Client('fair-myth-274206') # ENTER CLIENT ID

        # parameters for loading data and images
        face_detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
        leg_detection_model_path = 'haarcascades/haarcascade_lowerbody.xml'
        emotion_model_path = 'models/emotion_model.hdf5'

        # hyper-parameters for bounding boxes shape
        # loading models
        self.face_detection = cv2.CascadeClassifier(face_detection_model_path)
        self.leg_detection = cv2.CascadeClassifier(leg_detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

        self.camera_uri = camera_url
        self.total_bytes = b''
        self.stream = urlopen(camera_url)

        # cv2.namedWindow("main")
        # self.camera = cv2.VideoCapture(0)

    def getPicture(self):
        self.total_bytes += self.stream.read(1024)
        b = self.total_bytes.find(b'\xff\xd9') # JPEG end
        if not b == -1:
            a = self.total_bytes.find(b'\xff\xd8') # JPEG start
            jpg = self.total_bytes[a:b+2] # actual image
            self.total_bytes = self.total_bytes[b+2:] # other informations
            
            # decode to colored image ( another option is cv2.IMREAD_GRAYSCALE )
            img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR) 
            return img, True
        return None, False

    def start(self):
        while True:
            # frame = camera.read()[1]
            frame, ok = self.getPicture()
            if not ok:
                continue

            frame = imutils.resize(frame,width=400)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            legs = self.leg_detection.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            frameClone = frame.copy()

            if len(faces) > 0:
                faces = sorted(faces, reverse = True,
                key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = self.emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = self.EMOTIONS[preds.argmax()]
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                (0, 0, 255), 2)
                data = {
                    u'timestamp' : firestore.SERVER_TIMESTAMP,
                    u'emotion': str(label),
                    u'width': str(fW),
                    u'height': str(fH)
                }
                self.store.collection(u'Face').document().set(data, merge=True )
                print("faces," + label+","+str(fW)+","+str(fH))

            if len(legs) > 0:
                legs = sorted(legs, reverse = True,
                key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = legs
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                cv2.putText(frameClone, "legs", (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                (0, 0, 255), 2)
                data = {
                    u'timestamp' : firestore.SERVER_TIMESTAMP,
                    u'width': str(fW),
                    u'height': str(fH)
                }
                self.store.collection(u'Legs').document().set(data, merge=True )                
                print("legs,"+"none,"+str(fW)+","+str(fH))
            # else:
                # cv2.imshow('main', frameClone)

        # camera.release()
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    rt = RealTimeRecognition(sys.argv[1])
    rt.start()
    
