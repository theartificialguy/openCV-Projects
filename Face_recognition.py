# # Face detection and saving faces
import numpy as np
import cv2
import pickle

face_classifier = cv2.CascadeClassifier("F:/haarcascade/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")
cap = cv2.VideoCapture(0)
labels1 = {"person_name":1}
with open("labels.pickle","rb") as f:
    og_labels = pickle.load(f)
    labels1 = {v:k for k,v in og_labels.items()}

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.5,5)
    for x,y,w,h in faces:
        #cv2.rectangle(frame,(x,y),(x+h,y+w),color='r',3)
        roi_gray = gray[y:y+h,x:x+w] #(y-start to y-end, x-start to x-end)
        #roi_color = frame[y:y+h,x:x+w]
        face_id, conf = recognizer.predict(roi_gray)
        if conf>=45:
            #print(face_id)
            cv2.putText(frame,labels1[face_id]+" "+str(round(conf,2))+"%",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
        face_item = "my_face.png"
        cv2.imwrite(face_item,roi_gray)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1)&0xff
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()


