# pip install opencv-contrib-python --upgrade
# or without extra modules
# pip install opencv-python 
# C:\> python
# >>> import cv2
# >>> print(cv2.__version__)
# >>> print(cv2.__file__)
# pip install --upgrade
# https://github.com/codingforentrepreneurs/OpenCV-Python-Series
# D:\Coder\Python\face\facerg\lib\site-packages\cv2\

import numpy as np
import cv2
import pickle

# cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# cap.set(10,100)

# while True:
#     success, img = cap.read()
#     cv2.imshow("Video",img)
#     if cv2.waitKey(1) & 0xFF ==ord ('q'):
#         break

face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')    
recognizer  = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("label.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# cap.set(10,100)
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

# make_720p()
# change_res(1280, 720)

# def rescale_frame(frame, percent=75):
#     width = int(frame.shape[1] * percent/ 100)
#     height = int(frame.shape[0] * percent/ 100)
#     dim = (width, height)
#     return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    # rect, frame = cap.read()
    # frame75 = rescale_frame(frame, percent=30)
    # cv2.imshow('frame75', frame75)
    # frame150 = rescale_frame(frame, percent=45)
    # cv2.imshow('frame150', frame150)

# while True: 
    # capture frame by frame
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        # roi = region of interest
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycard_end)
        roi_color = frame[y:y+h, x:x+w] 

        # recognize ? deep learned model predict keras tensorflow pyttorch scikit learn
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45: # and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0) #BGR 0-255
        stroke = 2
        # width = x + w
        end_cord_x = x + w
        # height = y + h
        end_cord_y = y + h
        # cv2.rectangle(frame, (x, y), (width, height))
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    # cv2.imshow('abu',gray)
    cv2.imshow('frame1',frame)
    # cv2.imshow('frame2',frame)
    # cv2.imshow('frame3',frame)



    if cv2.waitKey(1) & 0xFF ==ord ('q'):
        break

# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# end of code