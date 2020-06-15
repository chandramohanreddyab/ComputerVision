import sys
# Telling the system how human face looks like
import cv2

cascadeFitler = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# defining object to open front camera
videoCapture = cv2.VideoCapture(0)

while True:
    # continous flow of video frame
    ret, frame = videoCapture.read()
    # the filter recognizes the images in the gray scale format
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # take a frame of image then identify the face and return the faces in list
    faces = cascadeFitler.detectMultiScale(
        grayImage,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # now we have faces, so highlight them in rectangular box

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # now diplay the frame with rectangles
    cv2.imshow('FaceDetectionFrame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
videoCapture.release()