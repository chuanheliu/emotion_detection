## 说明

dlib+SVM 情绪识别第一版，只为留个底。识别率不高的一个demo，都是调用的存在的库，后续工作要大改。

训练集用的ck，放在/train_image文件夹内。

子文件为：

/train_image/0/ (label为0的图像)

...

/train_image/6/ (label为6的图像)

## 环境：

Python2.7 

Opencv

dlib

sklearn



## 运行：

##### 训练：

`python train.py`

##### 使用：(可以直接使用训练好的模型train_model.m)

调摄像头识别

`python camera_predict.py` 

识别图片中人脸表情

`python image_predict.py`

## 附件

没有上传.shape_predictor_68_face_landmarks.dat