# -*- coding: utf-8 -*-

"""
Created on Thu Apr 06 20:52:04 2017

@author: Administrator
"""
import dlib
import cv2
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def find_face_landmarks(img):
    time0 = time.time()
    points = []
    rects = detector(img,0)
    print img
    time1 = time.time()
    print ((time1 - time0)*1000,'ms')
    if len(rects) == 0:
        return [],points

    shape = predictor(img,rects[0])
    time2 = time.time()
    print ((time2 - time1)*1000,'ms')
    for i in range(0,68):
        points.append((shape.part(i).x,shape.part(i).y))

    return rects[0],points

def draw_face_landmarks(img,rect,landmarks):
    img_dst = img.copy()
    cv2.rectangle(img_dst,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(255,0,0),2)
    for i in range(0,68):
        cv2.circle(img_dst,points[i],2,(0,0,255),-1) #-1表示填充

    return img_dst

video = cv2.VideoCapture(0)

print (video.isOpened())
if video.isOpened():
    print ('width: %d' % video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print ('height: %d' % video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    success,frame = video.read()

    while success:
        rect,points = find_face_landmarks(frame)
        if not rect:
            success,frame = video.read()
            continue
        img = draw_face_landmarks(frame,rect,points)

        cv2.imshow('face_landmarks',img)
        cv2.waitKey(1)

        success,frame = video.read()
    video.release()
    cv2.destroyAllWindows()