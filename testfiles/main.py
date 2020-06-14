import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

# img = cv2.imread('/Users/Anirudh/Desktop/test.jpg',cv2.IMREAD_COLOR)
# img = imutils.resize(img,width=400)

# face_cascade = cv2.CascadeClassifier('/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
# low_cascade = cv2.CascadeClassifier('/opt/local/share/OpenCV/haarcascades/haarcascade_lowerbody.xml')



# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )

# faces = face_cascade.detectMultiScale(gray, 1.1 , 4)
# low = low_cascade.detectMultiScale(gray, 1.1 , 3)
    
# for (x,y,w,h) in faces:
#     cv2.rectangle(img, (x,y), (x+w, y+h), (12,150,100),2)
# # for (x,y,w,h) in low:
# #     cv2.rectangle(img, (x,y), (x+w, y+h), (12,150,100),2)
    
# cv2.imshow('image',img)
# cv2.waitKey(0) # If you don'tput this line,thenthe image windowis just a flash. If you put any number other than 0, the same happens.
# cv2.destroyAllWindows()

cv2.namedWindow('frame')
cv2.namedWindow('dist')

# the classifier that will be used in the cascade
#HAAR FACE CASCADE
#faceCascade = cv2.CascadeClassifier('/Users/nkumar/CSCode/Private Projects/ObjDetectTest/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#LBP FACE CASCADE
faceCascade = cv2.CascadeClassifier('/Users/nkumar/BruinBot-Vision-AI/lbpcascades/lbpcascade_frontalface_improved.xml')
#HAAR UPPER CASCADE
#faceCascade = cv2.CascadeClassifier('/Users/nkumar/CSCode/Private Projects/ObjDetectTest/opencv/data/haarcascades/haarcascade_upperbody.xml')
low_cascade = cv2.CascadeClassifier('/Users/nkumar/BruinBot-Vision-AI/haarcascades/haarcascade_lowerbody.xml')

#capture video stream from camera source. 0 refers to first camera, 1 referes to 2nd and so on.
cap = cv2.VideoCapture(0)


triggered = False
sdThresh = 5
font = cv2.FONT_HERSHEY_SIMPLEX

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
while(True):
    _, frame3 = cap.read()
    frame3 = imutils.resize(frame3, width=600)
    rows, cols, _ = np.shape(frame3)
    cv2.imshow('dist', frame3)
    dist = distMap(frame1, frame3)

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

    # begin lower  cascade
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    lowers = low_cascade.detectMultiScale(
        gray,
        scaleFactor=2,
        minSize=(10, 10),
        minNeighbors=2
    )
    facecount = 0
    # draw a rectangle over detected lowerbody
    for (x, y, w, h) in lowers:
        facecount = facecount + 1
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 1)
        print('legs')

    if stDev > sdThresh:
            # the cascade is implemented in grayscale mode
            gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # begin face cascade
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.5,
                minSize=(20, 20),
                minNeighbors=1
            )
            facecount = 0
            # draw a rectangle over detected faces
            for (x, y, w, h) in faces:
                facecount = facecount + 1
                cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 1)
                print('face')
            cv2.putText(frame2, "No of faces {}".format(facecount), (50, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    else:
            if facecount > 0:
                    print("Face count:")
                    print(facecount)
                    facecount = 0
    cv2.imshow('frame', frame2)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
