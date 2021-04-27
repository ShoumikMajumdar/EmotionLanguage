import numpy as np
import cv2 as cv
import dlib

def read_image():
    img = cv.imread("emotion_test_images/2.jpg")
    return img

def preprocess(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #img = cv.resize(img,(48,48))
    return img

def show(label,img):
    img = cv.resize(img,(360,400))
    cv.imshow(label,img)
    cv.waitKey(0)
    cv.destroyAllWindows()


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frame = read_image()
gray = preprocess(frame)
faces = detector(gray)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    landmarks = predictor(gray,face)
    for n in range(0,68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv.circle(frame,(x,y),3,(0,0,255),-1)
    
cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)

show("frame",frame)

