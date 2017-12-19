
# -*- coding: utf-8 -*-

from sklearn.externals import joblib
import call_dlib as cd
import numpy as np
import  cv2


clf = joblib.load("train_model.m")

def predict(path):

    react,points = cd.find_face_landmarks(cv2.imread(path))

    p = np.array(points).reshape(1,-1)
    print(p)
    print clf.predict(p)


for i in range(10):
    predict('/home/chuanhe/PycharmProjects/svm_face/test/2/'+ str(i+1)+ '.png')
    # predict('1.png')