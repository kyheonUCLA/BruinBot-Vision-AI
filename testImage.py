from cmath import e
from concurrent.futures import process
from pyexpat import features
from xml.dom.minidom import Document
from cv2 import merge
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
import sys
import os

import google.cloud
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app

class ImageBotRecognition:
    # Static class variable accessible to every instance of the class
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"] 

    # Use opencv to read specified image, and creates a greyscale copy to be used in the detection algorithm (YOLO V3)
    # The purpose of the greyscale copy is to minimize computation/runtime
    def __init__(self, img_path) -> None:
        (self.face_detection, self.leg_detection, self.emotion_classifier) = self.load_models()
        self.og_image = cv2.imread(img_path)
        if type(self.og_image) == type(None):
            print('Invalid image path was supplied')
        
        self.grey_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.image_path = img_path

    # returns a tuple of classifiers for each model (face, leg, emotion) based on the pretrained haarcascade models
    def load_models(self):
        face_detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
        leg_detection_model_path = 'haarcascades/haarcascade_lowerbody.xml'
        emotion_model_path = 'models/emotion_model.hdf5'
        face_detection = cv2.CascadeClassifier(face_detection_model_path)
        leg_detection = cv2.CascadeClassifier(leg_detection_model_path)
        emotion_classifier = load_model(emotion_model_path, compile=False)
        return (face_detection, leg_detection, emotion_classifier)
    

    # Uses image read in the constructor to detect emotion, and output the prediction as an image in the test_output2 folder 

    def run_in_local_mode(self):
        detected = False
        faces = self.face_detection.detectMultiScale(self.grey_image,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        try:
            roi, fX, fY, fW, fH = self.process_image(faces)
            predictions = self.emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(predictions)
            detected_emotion = ImageBotRecognition.EMOTIONS[predictions.argmax()]
            cv2.putText(self.og_image, detected_emotion, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(self.og_image, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
            labelStr = str(detected_emotion)
            widthStr = str(fW)
            heightStr = str(fH)
            print("faces," + labelStr +"," + widthStr +"," + heightStr)
            detected = True
            #cv2.imwrite('test_output2/' + self.image_path.split('/')[-1], self.og_image)
        except:
            print('No Face Detected')

        legs = self.leg_detection.detectMultiScale(self.grey_image,scaleFactor=2, minNeighbors=2,minSize=(10,10))
        try:
            roi, fX, fY, fW, fH = self.process_image(legs)
            cv2.putText(self.grey_image, "legs", (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(self.grey_image, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
            print("legs," + "none," + str(fW) + "," + str(fH))
            #cv2.imwrite('test_output/' + self.image_path.split('/')[-1], self.og_image)
            detected = True
        except:
            print("No Leg Detected")
        
        if detected:
            cv2.imwrite('test_output2/' + self.image_path.split('/')[-1], self.og_image)

    

    # Given a list of detected features, this function sorts the list based on the lambda.
    # The
    def process_image(self, feature):
        #print(len(features))
        if len(feature) > 0:
            feature = sorted(feature, reverse = True, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = feature
            roi = self.grey_image[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            return roi, fX, fY, fW, fH
        else:
            raise Exception


    # something is wrong with the store variable, and the permissions granted to admins
    def run_in_cloud_mode(self, credentials_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path # this line migth be unnecessary
        cred = credentials.Certificate(credentials_path)
        app = initialize_app(cred)
        db = firestore.client() # Client or client? 'fair-myth-274206'

        faces = self.face_detection.detectMultiScale(self.grey_image,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        try:            
            #print("faces," + labelStr +"," + widthStr +"," + heightStr)

            roi, fX, fY, fW, fH = self.process_image(faces)
            predictions = self.emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(predictions)
            detected_emotion = ImageBotRecognition.EMOTIONS[predictions.argmax()]
            cv2.putText(self.og_image, detected_emotion, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(self.og_image, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
            labelStr = str(detected_emotion)
            widthStr = str(fW)
            heightStr = str(fH)
            #firebase requires unicode encoded strings. Python3 strings are unicode by default
            fb_data = { 
                u'timestamp' : firestore.SERVER_TIMESTAMP,
                u'emotion': labelStr,
                u'width': widthStr,
                u'height': heightStr}
            
            data = db.collection('Faces')
            print(data)
            doc = data.document()
            print(doc)
            doc.set(fb_data, merge=True)
            #document().set(fb_data, merge=True )
            print("faces," + labelStr +"," + widthStr +"," + heightStr)
        except google.api_core.exceptions.PermissionDenied as e:
            print(e)
        except:
            print('No Face Detected')

        legs = self.leg_detection.detectMultiScale(self.grey_image,scaleFactor=2, minNeighbors=2,minSize=(10,10))
        try:
            roi, fX, fY, fW, fH = self.process_image(legs)
            cv2.putText(self.grey_image, "legs", (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(self.grey_image, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
            widthStr = str(fW)
            heightStr = str(fH)
            fb_data = { 
                'timestamp' : firestore.SERVER_TIMESTAMP,
                'width': widthStr,
                'height': heightStr}
            #db.collection('Legs').document().set(fb_data, merge=True)
            print("legs," + "none," + str(fW) + "," + str(fH))
        except:
            print("No Leg Detected")

if __name__ == "__main__":
    obj = ImageBotRecognition(sys.argv[1])
    #obj.run_in_cloud_mode('key.json')
    obj.run_in_local_mode()