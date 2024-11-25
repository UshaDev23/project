import os
from cv2 import cv2
import numpy as n
from PIL import Image



recognition = cv2.face.LBPHFaceRecognizer_create() # instantiate the lbph recognizer
path = '/Users/usha/Desktop/project/data/' #setting out path of folders with images
file='/Users/usha/Desktop/project/data/recognizer/'
if not os.path.exists(file):
    os.makedirs(file) # making a directory for yml file which will be generated after training

def getImageswithId(path):
    faces = []
    faceid = []

    for root,directory,filenames in os.walk(path):
        for filename in filenames:
            id = os.path.basename(root) #this directly assigns folder name ie 0,1..
            img_path = os.path.join(root,filename)
            print('img_path:',img_path)
            print('id:',id)
            test_img = cv2.imread(img_path)
           
            # test_img = np.float32(test_img)
            if test_img is None:
                print('image not loaded poperly - cv2 cant read!!')
                continue
            #if images in dataset are not in gray scale then use below 3 line

            gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
            face_haar_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#Load haar classifier
            face=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)#detectMultiScale returns rectangles
           
            #if len(face)!=1:
                #continue # since we are asuuming only single person images are being fed to classifier

            faces.append(face)
            faceid.append(id)
            cv2.destroyAllWindows()
    return faceid,faces


faceid , faces = getImageswithId(path)
recognition.train(faces, n.array(faceid))
recognition.write('/Users/usha/Desktop/project/data/recognizer/trainingData.yml')
























#https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b