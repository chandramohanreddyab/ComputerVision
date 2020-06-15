
import cv2,os
import numpy as np

#Data
DataSets='DataSets'
(width, height) = (1280,720) 
(images, lables, names, id) = ([], [], {}, 0) 
for dirs,subdirs,files in os.walk(DataSets):
    for subdir in subdirs:
        names[id] = subdir 
        img_path=os.path.join(DataSets, subdir)
        for filename in os.listdir(img_path): 
            path = img_path + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0)) 
            lables.append(int(lable)) 
        id += 1
(images, lables) = [np.array(lis) for lis in [images, lables]]


model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)


har_file='haarcascade_frontalface_default.xml'
cascadeClassifier=cv2.CascadeClassifier(har_file)


#Taking the input from WebCamera
webCam=cv2.VideoCapture(0)
while True:
    par,frame = webCam.read()
    gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=cascadeClassifier.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray_frame[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height)) 
        #predit the face and gives the label and confidence
        prediction = model.predict(face_resize) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 
        if prediction[0]<10:
            cv2.putText(frame, '% s - %.0f' % (names[prediction[0]], prediction[1]), (x-10, y-10),  cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            cv2.putText(frame, 'unknown',  (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
            
        cv2.imshow('FaceDetectionFrame', frame)
        key = cv2.waitKey(10)
        if key == 'q':
            break

cv2.destroyAllWindows()
webCam.release()





