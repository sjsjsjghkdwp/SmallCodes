# .ipynb 파일을 내용만 추출해 .py파일로 만든 것 입니다
# ln[1]
import warnings

import sys
import tensorflow as tf
import keras

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

from keras.models import load_model

import cv2
import numpy as np
import matplotlib.pyplot as plt

print('openCV version : ', cv2.__version__) 

# ln[2]
base = cv2.imread("severance_base.jpg")

# ln[3]
plt.figure(figsize=(70,50))
base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

ret, base_th = cv2.threshold(base_gray, 120, 230, cv2.THRESH_BINARY_INV)
plt.imshow(base_th);

# ln[4]
base1=base_th.copy()
contours_base, hierarchy_base= cv2.findContours(base1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
CA_base=np.array(range(len(contours_base)))
for i in range(len(contours_base)):
    CA_base[i]=cv2.contourArea(contours_base[i])
c_base=contours_base[CA_base.argmax()]

# ln[5]
LTmost_base=tuple(c_base[(c_base[:, :, 0]+c_base[:, :, 1]).argmin()][0])
RTmost_base=tuple(c_base[(c_base[:, :, 0]-c_base[:, :, 1]).argmax()][0])
RBmost_base=tuple(c_base[(c_base[:, :, 0]+c_base[:, :, 1]).argmax()][0])
LBmost_base=tuple(c_base[(c_base[:, :, 0]-c_base[:, :, 1]).argmin()][0])

"""
plt.figure(figsize=(70,50))
plt.imshow(base, cmap='bone')
plt.axis("off")
plt.scatter(
    [LTmost_base[0], RBmost_base[0], RTmost_base[0], LBmost_base[0]],
    [LTmost_base[1], RBmost_base[1], RTmost_base[1], LBmost_base[1]], 
    c="b", s=200)
plt.title("Extreme Points")

plt.show()
"""

# ln[6]
data1 = cv2.imread("severance_data1.jpg")

# ln[7]
data1_gray = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
ret, data1_th = cv2.threshold(data1_gray, 120, 230, cv2.THRESH_BINARY_INV)
plt.figure(figsize=(70,50))
plt.imshow(data1_th);

# ln[8]
data1c=data1_th.copy()
contours_data1, hierarchy_data1= cv2.findContours(data1c,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
CA_data1=np.array(range(len(contours_data1)))
for i in range(len(contours_data1)):
    CA_data1[i]=cv2.contourArea(contours_data1[i])
c_data1=contours_data1[CA_data1.argmax()]

# ln[9]
LTmost_data1=tuple(c_data1[(c_data1[:, :, 0]+c_data1[:, :, 1]).argmin()][0])
RTmost_data1=tuple(c_data1[(c_data1[:, :, 0]-c_data1[:, :, 1]).argmax()][0])
RBmost_data1=tuple(c_data1[(c_data1[:, :, 0]+c_data1[:, :, 1]).argmax()][0])
LBmost_data1=tuple(c_data1[(c_data1[:, :, 0]-c_data1[:, :, 1]).argmin()][0])

"""
plt.figure(figsize=(70,50))
plt.imshow(data, cmap='bone')
plt.axis("off")
plt.scatter(
    [LTmost_data[0], RBmost_data[0], RTmost_data[0], LBmost_data[0]],
    [LTmost_data[1], RBmost_data[1], RTmost_data[1], LBmost_data[1]], 
    c="b", s=200)
plt.title("Extreme Points")

plt.show()
"""

# ln[10]
pts1 = np.float32([LTmost_data1, LBmost_data1, RTmost_data1, RBmost_data1])

    # 좌표의 이동점
pts2 = np.float32([LTmost_base, LBmost_base, RTmost_base, RBmost_base])

M = cv2.getPerspectiveTransform(pts1, pts2)

base_shape=np.shape(base_gray)
a=base_shape[1]
b=base_shape[0]
dst1 = cv2.warpPerspective(data1, M, (a,b))

plt.figure(figsize=(70,50)),plt.subplot(121),plt.imshow(data1),plt.title('image')
plt.figure(figsize=(70,50)),plt.subplot(122),plt.imshow(dst1),plt.title('Perspective')
plt.show()

# ln[11]
real1=cv2.subtract(base,dst1)
plt.figure(figsize=(70,50))
plt.imshow(real1)

# ln[12]
real1_gray = cv2.cvtColor(real1, cv2.COLOR_BGR2GRAY)
ret, real1_th = cv2.threshold(real1_gray, 200, 230, cv2.THRESH_BINARY_INV)
plt.figure(figsize=(70,50))
plt.imshow(real1_th)

# ln[13]
plt.figure(figsize=(70,50)),plt.subplot(121),plt.imshow(base)
plt.figure(figsize=(70,50)),plt.subplot(122),plt.imshow(dst2)

# ln[14]
data2 = cv2.imread("severance_data2.jpg")

data2_gray = cv2.cvtColor(data2, cv2.COLOR_BGR2GRAY)
ret, data2_th = cv2.threshold(data2_gray, 120, 230, cv2.THRESH_BINARY_INV)

data2c=data2_th.copy()
contours_data2, hierarchy_data2= cv2.findContours(data2c,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
CA_data2=np.array(range(len(contours_data2)))
for i in range(len(contours_data2)):
    CA_data2[i]=cv2.contourArea(contours_data2[i])
c_data2=contours_data2[CA_data2.argmax()]

LTmost_data2=tuple(c_data2[(c_data2[:, :, 0]+c_data2[:, :, 1]).argmin()][0])
RTmost_data2=tuple(c_data2[(c_data2[:, :, 0]-c_data2[:, :, 1]).argmax()][0])
RBmost_data2=tuple(c_data2[(c_data2[:, :, 0]+c_data2[:, :, 1]).argmax()][0])
LBmost_data2=tuple(c_data2[(c_data2[:, :, 0]-c_data2[:, :, 1]).argmin()][0])

pts2_1 = np.float32([LTmost_data2, LBmost_data2, RTmost_data2, RBmost_data2])
pts2_2 = np.float32([LTmost_base, LBmost_base, RTmost_base, RBmost_base])

M = cv2.getPerspectiveTransform(pts2_1, pts2_2)

dst2 = cv2.warpPerspective(data2, M, (a,b))

real2=cv2.subtract(base,dst2)
plt.figure(figsize=(70,50))
plt.imshow(real2)

real1_gray = cv2.cvtColor(real1, cv2.COLOR_BGR2GRAY)
ret, real1_th = cv2.threshold(real1_gray, 200, 230, cv2.THRESH_BINARY_INV)
plt.figure(figsize=(70,50))
plt.imshow(real1_th)