import os
import numpy as np 
from PIL import Image
import pickle
import cv2
# Grabbing path and labels for images
face_classifier = cv2.CascadeClassifier("F:/haarcascade/haarcascade_frontalface_default.xml")
x_train = []
y_labels = []
current_id = 0
labels_id = {}
base_dir = os.path.dirname(os.path.abspath('__file__')) # will give the current location of directory 
                                                        #in which we are
img_dir = os.path.join(base_dir,"images")
for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith(".png") or file.endswith(".jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            #print(label, path)
            if not label in labels_id:
                labels_id[label] = current_id
                current_id += 1
            id_ = labels_id[label]
            #x_train.append(path)#turn into np array (gray)
            #y_labels.append(label)#need to convert into some number
            #converting path(png files)into np array and labels into numbers
            pil_img = Image.open(path).convert("L") # convert to grayscale
            final_img = pil_img.resize((550,550),Image.ANTIALIAS)
            img_array = np.array(final_img,dtype='uint8')
            #print(img_array)
            #now we need only the face roi of these nunmpy arrays (numpy images)
            #face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            faces2 = face_classifier.detectMultiScale(img_array,1.5,5)
            for (x,y,w,h) in faces2:
                roi = img_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
            #print(x_train)
            #print(y_labels)
print(labels_id)
#To read/load any trainer that you created use: recognizer.read("your_trainer_name")
#To save these labels and ids
with open("labels.pickle",'wb') as f:
    pickle.dump(labels_id,f)
#Building opencv recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(x_train,np.array(y_labels))
recognizer.save("Trainer.yml")