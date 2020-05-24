from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

cv2.namedWindow('main')
cv2.namedWindow('dist') #remove soon

EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised" ,"neutral"]
faceCascade = cv2.CascadeClassifier('./lbpcascades/lbpcascade_frontalface_improved.xml')
legCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_lowerbody.xml')
emotion_model_path = '/Users/nkumar/BruinBot-Vision-AI/models/_mini_XCEPTION.106-0.65.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)

cap = cv2.VideoCapture(0)
triggered = False
sdThresh = 5
font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist

_, frame1 = cap.read()
_, frame2 = cap.read()
facecount = 0
frame1 = imutils.resize(frame1, width=600)
frame2 = imutils.resize(frame2, width=600)

#def blank():
    

while True:
    _, frame3 = cap.read()
    #reading the frame
    frame3= imutils.resize(frame3,width=600)
    rows, cols, _ = np.shape(frame3)
    cv2.imshow('dist', frame3)
    dist = distMap(frame1, frame3)
    #print("dist="+str(dist))

    frame1 = frame2
    frame2 = frame3

    # apply Gaussian smoothing
    mod = cv2.GaussianBlur(dist, (9,9), 0)

    # apply thresholding
    _, thresh = cv2.threshold(mod, 100, 255, 0)

    # calculate st dev test
    _, stDev = cv2.meanStdDev(mod)

    cv2.imshow('dist', mod)
    cv2.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=1,minSize=(20,20),flags=cv2.CASCADE_SCALE_IMAGE)
    frameClone = frame3.copy() 
    
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    lowers = legCascade.detectMultiScale(
        gray,
        scaleFactor=2,
        minSize=(10, 10),
        minNeighbors=2
    )
    for (x, y, w, h) in lowers:
        facecount = facecount + 1
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 1)
        print('legs')

    gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    if stDev > sdThresh:
        
        facecount = 0

        for (x, y, w, h) in faces:
            facecount = facecount + 1
            cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 1)
            print("x="+str(x)+"y="+str(y)+"w="+str(w)+"h="+str(h))
            print('face')

        cv2.putText(frame2, "No of faces {}".format(facecount), (50, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

        canvas = np.zeros((250, 300, 3), dtype="uint8")
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
                        # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
                # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            print(label)
        else: continue

    
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                (0, 0, 255), 2)
    else:
        if facecount > 0:
            print("Face Count:")
            print(facecount)
            facecount = 0

    cv2.imshow('main', frameClone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()