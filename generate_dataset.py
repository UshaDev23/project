from cv2 import cv2
import os
def dataset():
    face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #file_name=input("Enter name:")
    path='/Users/usha/Desktop/project/data/'
    i=0
    for files in os.walk(path):
                file_name = '{}'.format(i)
                file=os.path.join(path,file_name)
                if not os.path.exists(file):
                     os.mkdir(file)
                else:
                    i=i+1
    def face_crop(img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)
        if faces is():
           return None
        for(x,y,w,h) in faces:
            offset=5
            cropped_face=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        return cropped_face
    cap=cv2.VideoCapture(0)
    img_id=0
 
    while(True):
        ret,frame=cap.read()
        if face_crop(frame) is not None:
            img_id+=1
            face=cv2.resize(face_crop(frame),(500,500))
            face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            #file_path='file/'+str(img_id)+".jpg"
            #file_path="data/"+file_name+"_"+str(img_id)+".jpg"
            file_path=os.path.join(file,"{}.jpg").format(str(img_id))
            cv2.imwrite(file_path,face)
            cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow("Cropped face",face)
            if cv2.waitKey(1)==13 or int(img_id)==60:
                break
    cap.release()
    cv2.destroyAllWindows()
dataset()

