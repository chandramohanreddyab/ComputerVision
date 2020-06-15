import numpy as np
# Importing the cv and Keras for model training
import cv2
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Dropout, MaxPool2D,  Flatten
from emotion_detection_model import emotion_detection,model

num_classes = 7
cascade_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video=cv2.VideoCapture(0)
emotion_dict = {0: "Hey,Are you angry with me ?", 1: " You are Disgusted", 2: "Dont be Fearful", 3: "You are Happy and pretty", 4: "Please smile", 5: "Why are you Sad ?", 6: "Damn.! You are Surprised"}


model=emotion_detection('load weights',model)

while True:
    ret,frame=video.read()

    gray_impage=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # take a frame of image then identify the face and return the faces in list
    faces = cascade_classifier.detectMultiScale( gray_impage,scaleFactor=1.2,minNeighbors=5,minSize=(25, 25),flags=cv2.CASCADE_SCALE_IMAGE)
    #take each face and rectangle box
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray_impage[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow('Emotion Detection',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
