
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 06 20:52:04 2017

@author: chuanhe
"""
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def find_face_landmarks(img):

    points = []

    rects = detector(img,0)

    if len(rects) == 0:
        return [],points
    shape = predictor(img,rects[0])

    for i in range(0,68):
        points.append((int(shape.part(i).x),int(shape.part(i).y)))


    return rects[0],points


def draw_face_landmarks(img,rect,points):
    img_dst = img.copy()
    cv2.rectangle(img_dst,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(255,0,0),2)
    for i in range(0,68):
        cv2.circle(img_dst,points[i],2,(0,0,255),-1) #-1表示填充
    return img_dst, points






