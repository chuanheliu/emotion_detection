# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from sklearn import svm
from sklearn import cross_validation
import call_dlib as cd
import numpy as np
import os
import cv2

def all_path(dirname, write_path):


    if os.path.isfile("label.txt"):
        os.remove("label.txt")

    file_list = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            file_list.append(apath)
            if ("/0/" in apath):
                write_file(write_path, apath + " 0\n")
            elif ("/1/" in apath):
                write_file(write_path, apath + " 1\n")
            elif ("/2/" in apath):
                write_file(write_path, apath + " 2\n")
            elif ("/3/" in apath):
                write_file(write_path, apath + " 3\n")
            elif ("/4/" in apath):
                write_file(write_path, apath + " 4\n")
            elif ("/5/" in apath):
                write_file(write_path, apath + " 5\n")
            elif ("/6/" in apath):
                write_file(write_path, apath + " 6\n")
            else:
                print("Error: " + apath)
    return file_list


def write_file(path, text):
    file = open(path + "/label.txt", 'a+')
    file.write(text)


def path_label():

    print 'Pre-treatment: get image path and the label...'
    open_path = "/home/chuanhe/PycharmProjects/svm_face/train_image"
    white_path = "/home/chuanhe/PycharmProjects/svm_face"
    all_path(open_path, white_path)
    print "Done!"

def training_data(path):

    points_list = []
    label_list = []

    # 每次开始先删除文件
    if os.path.isfile("points.txt"):
        os.remove("points.txt")

    print 'Read training image...it will take few minutes'
    for line in open(path):
        # print line[:-3]
        img = cv2.imread(line[:-3])

        rect, points = cd.find_face_landmarks(img)

        #追加写文件
        file = open("points.txt", 'a+')
        file.write(line[:-3]+'\n'+str(points) + ' ' + str(int(line[-2:-1]))+'\n')

        points_list.append(np.array(points).reshape(1,-1)[0])
        label_list.append(int(line[-2:-1]))


    return points_list,label_list


if __name__ == '__main__':

    path_label()

    X,y = training_data('label.txt')

    target = np.array(y)
    train = np.array(X)


    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, target,test_size=0.4,random_state=0)

    print 'Training...'
    clf = svm.SVC(kernel='poly').fit(X_train, y_train)
    print 'Training success'

    print 'Save model'

    if os.path.isfile("train_model.m"):
        os.remove('train_model.m')
    joblib.dump(clf, "train_model.m")

    print ('Correctness rate: ' + str(clf.score(X_test, y_test)))

