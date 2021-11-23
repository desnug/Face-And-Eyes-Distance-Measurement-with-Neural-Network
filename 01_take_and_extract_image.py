import cv2
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from csv import writer

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

sample_image = 5
counter = 1
distance = 30
dir_name = 'E:/MyDrive'

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.2, flags=cv2.CASCADE_SCALE_IMAGE)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # cv2.putText(frame, 'face: '+str(w)+','+str(h), (10, 30), font, 0.6, (255, 255, 255), 2)
        eyes = eyeCascade.detectMultiScale(roi_gray)
        # Draw a rectangle around the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # cv2.putText(frame, 'eyes: '+str(ew)+','+str(eh),(10, 50), font, 0.6, (255, 255, 255), 2)

    if cv2.waitKey(1) & 0xFF == ord('c'):

        cv2.imwrite(f'{dir_name}{distance}.{counter}.jpg', frame_copy)
        print(f'capture image->{counter}, with distance: {distance} cm')
        counter += 1
        if counter > sample_image:
            break

    # Display the resulting frame
    cv2.imshow('Take Image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

# Get list of all files only in the given directory
imfilelist = [os.path.join(dir_name, f)
              for f in os.listdir(dir_name) if f.endswith(".jpg")]
#list_of_files = filter( lambda x: os.path.isfile(os.path.join(dir_name, x)), os.listdir(dir_name) )

# Sort list of files based on last modification time in ascending order
list_of_files = sorted(
    imfilelist, key=lambda x: os.path.getmtime(os.path.join(dir_name, x)))

# append data to csv file
with open('dataset_pixel.csv', 'a', newline='') as f_object:
    # Pass this file object to csv.writer()
    writer_object = writer(f_object)
    for el in list_of_files:
        imagen = cv2.imread(el)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # To see time added file
        for file_name in list_of_files:
            file_path = os.path.join(dir_name, file_name)
            timestamp_str = time.strftime('%m/%d/%Y :: %H:%M:%S',
                                          time.gmtime(os.path.getmtime(file_path)))
        # Detect faces
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.2, flags=cv2.CASCADE_SCALE_IMAGE)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = imagen[y:y+h, x:x+w]
            roi_color = imagen[y:y + h, x:x + w]

        # Detect eyes
        eyes = eyeCascade.detectMultiScale(roi_gray)
        # Draw a rectangle around the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(imagen, '', (x + ex, y + ey), 1, 1, (0, 255, 0), 1)

        data = str(w)+','+str(h)+','+str(ew)+','+str(eh)+','+str(distance)

        writer_object.writerow([w, h, ew, eh, distance])

        print(data)
