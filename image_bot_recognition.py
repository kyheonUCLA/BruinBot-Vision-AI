from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
import sys

import google.cloud
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("./key.json") #ENTER PATH LENGTH TO CREDENTIALS HERE
app = firebase_admin.initialize_app(cred)

store = firestore.Client('fair-myth-274206')


# parameters for loading data and images
face_detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
leg_detection_model_path = 'haarcascade_files/haarcascade_lowerbody.xml'
emotion_model_path = 'models/emotion_model.hdf5'
img_path = sys.argv[1]

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(face_detection_model_path)
leg_detection = cv2.CascadeClassifier(leg_detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

orig_frame = cv2.imread(img_path)
frame = cv2.imread(img_path,0)

faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
legs = leg_detection.detectMultiScale(frame,scaleFactor=2, minNeighbors=2,minSize=(10,10))


if len(faces) > 0:
    faces = sorted(faces, reverse = True,
    key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    cv2.putText(orig_frame, label, (fX, fY - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),
                    (0, 0, 255), 2)
    labelStr = str(label)
    widthStr = str(fW)
    heightStr = str(fH)
    data = {
       u'timestamp' : firestore.SERVER_TIMESTAMP,
       u'emotion': str(label),
       u'width': str(widthStr),
       u'height': str(heightStr)
    }
    store.collection(u'Face').document().set(data, merge=True )
    #store.collection(u'Face').order_by(u'timestamp',u'asc')
    print("faces," + label+","+str(fW)+","+str(fH))
    cv2.imwrite('test_output/'+img_path.split('/')[-1],orig_frame)
else:
    cv2.imshow('main', orig_frame)

if len(legs) > 0:
    legs = sorted(legs, reverse = True,
    key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = legs
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    cv2.putText(orig_frame, "legs", (fX, fY - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),
                    (0, 0, 255), 2)
    data = {
       u'timestamp' : firestore.SERVER_TIMESTAMP,
       u'width': unicode(widthStr),
       u'height': unicode(heightStr)
    }
    store.collection(u'Legs').document().set(data, merge=True )
    #store.collection(u'Legs').order_by(u'timestamp',u'asc')
    print("legs,"+"none,"+str(fW)+","+str(fH))
    cv2.imwrite('test_output/'+img_path.split('/')[-1],orig_frame)
else:
    cv2.imshow('main', orig_frame)

cv2.imshow('main', orig_frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    sys.exit("Processed")


cv2.destroyAllWindows()
