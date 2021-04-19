import cv2 as cv
import numpy as np
import sys
import torch
from torchvision import transforms
from VGG import VGG
#from dataset import FER2013
from PIL import Image
from utils import eval, detail_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

def predict(model,x):
    crop_size = 44
    transform_test = transforms.Compose([
            transforms.TenCrop(crop_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])

    roi_gray = Image.fromarray(np.uint8(x))
    inputs = transform_test(roi_gray)
    ncrops, c, ht, wt = np.shape(inputs)
    inputs = inputs.view(-1, c, ht, wt)
    inputs = inputs.to(device)
    outputs = model(inputs)
    outputs = outputs.view(ncrops, -1).mean(0)
    _, predicted = torch.max(outputs, 0)
    expression = classes[int(predicted.cpu().numpy())]
    return expression


def show(label,img):
    img = cv.resize(img,(400,300))
    cv.imshow(label,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def read_image():
    img = cv.imread("2.jpg")
    return img
    
def preprocess(img,crop_size):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.resize(img,(48,48))
    return img

def main():
    #video_capture = cv.VideoCapture(0)        
    # while True:
    #     _ , img = video_capture.read()
    #     grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    crop_size= 44
    model = VGG()
    checkpoint = torch.load("model_state.pth.tar")
    model.load_state_dict(checkpoint['model_weights'])
    model.to(device)
    model.eval()


    face_detector= cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = read_image()
    grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey, 1.3, 6)
    
    for face in faces:
        x,y,w,h = face
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
        crop_img = img[y:y+h,x:x+w].copy()
        crop_img = preprocess(crop_img,crop_size)
        expression = predict(model,crop_img)
        cv.putText(img, expression, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 5)
        
    show("faces",img)


if __name__ == "__main__":
    main()