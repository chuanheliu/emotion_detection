
# -*- coding: utf-8 -*-

from sklearn.externals import joblib
import call_dlib as cd
import numpy as np


import cv2

clf = joblib.load("train_model.m")

#
# react,points = cd.find_face_landmarks(cv2.imread('5.jpeg'))
#
# p = np.array(points).reshape(1,-1)
# print(p)
# print clf.predict(p)
#



video = cv2.VideoCapture(1)

if video.isOpened():

    success,frame = video.read()

    while success:
        rect,points = cd.find_face_landmarks(frame)

        if not rect:
            success,frame = video.read()
            continue

        label = clf.predict(np.array(points).reshape(1,-1))
        if label == 0:
            print "0: happy"
        if label == 1:
            print "1: disgust"
        if label == 2:
            print "2: fear"
        if label == 3:
            print "3: suprise"
        if label == 4:
            print "4: angry"
        if label == 5:
            print "5: sad"
        if label == 6:
            print "6: neural"



        img,points = cd.draw_face_landmarks(frame,rect,points)
        cv2.imshow('face_landmarks',img)
        cv2.waitKey(1)

        success,frame = video.read()
    video.release()
    cv2.destroyAllWindows()

