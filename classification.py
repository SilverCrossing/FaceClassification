import sklearn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.io import loadmat
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 将rawdata转为图片
# print(os.listdir("./face/face/rawdata"))   # 读取文件路径
# list1 = []   # 用于存放图片尺寸
# for filename in os.listdir("./face/face/rawdata"):   # 读取文件夹内的所有图片
#     with open("./face/face/rawdata/"+filename, "rb") as file2:   # 读取文件
#         img_data = file2.read()
#     img_data = np.frombuffer(img_data, dtype=np.uint8)   # 将文件换出来
#     if len(img_data) == 16384:   # 只用到128*128的图片，其他的都不用，因此只转换128的
#         img = img_data.reshape(128, 128)   # 转为128*128
#     else:
#         print("未知图片大小：", len(img_data))
#     cv2.imwrite("./face/face/picture/"+filename+".jpg", img)
#     if len(img_data) not in list1:
#         list1.append(len(img_data))
# print("list1:", list1)


# 导入mat文件
# face_mat_file = loadmat("./face/face/eigenfaces/readImage.mat")
# ev_mat_file = loadmat("./face/face/eigenfaces/ev.mat")
# print("readImage:", face_mat_file)
# print("ev:", ev_mat_file)
# print(type(face_mat_file), len(face_mat_file))
# print(type(ev_mat_file), len(ev_mat_file))
# data1 = ev_mat_file["eigenfaces"]
# print(data1.shape)
# for i in range(0, 99):
#     img_data = data1[i].reshape(128, 128)


# 因为数据太少，因此采用数据增强，将图片镜像从而扩大训练集
# def flip(image):
#     flipped_image = np.fliplr(image)
#     return flipped_image
#
#
# file_dir = "./face/face/glasses/"
# for img_name in os.listdir(file_dir):
#     img_path = file_dir + img_name
#     img = cv2.imread(img_path)
#     # 镜像
#     flipped_img = flip(img)
#     cv2.imwrite(file_dir + img_name[0:-4] + '_fli.jpg', flipped_img)


def img2vec(image, blocksize):
    image_w, image_h = 128, 128
    # 图像方块数据转换为向量
    img_block_vec = np.zeros((int(image_w * image_h / blocksize / blocksize), blocksize * blocksize))
    i = 0
    for r in range(0, image.shape[0], blocksize):
        for c in range(0, image.shape[1], blocksize):
            block = image[r:r + blocksize, c:c + blocksize]
            block = np.resize(block, (1, blocksize * blocksize))
            img_block_vec[i] = block
            i += 1
    return img_block_vec


def vec2img(vectors, blocksize):
    # 向量数据转换为图像
    img_kl = np.zeros((128, 128))
    i = 0
    for r in range(0, img_kl.shape[0], blocksize):
        for c in range(0, img_kl.shape[1], blocksize):
            block = vectors[i]
            block = np.resize(block, (blocksize, blocksize))
            i += 1
            img_kl[r:r + blocksize, c:c + blocksize] = block
    return img_kl.astype(int)


def kl_transform(vectors, principal_n):
    # 计算协方差矩阵和特征
    cov_matrix = np.cov(vectors.T)
    _, fvec = np.linalg.eig(cov_matrix)  # 输出的特征值默认降序排列
    img_kl_block_vec = np.dot(vectors, fvec)
    # 压缩，将非前N个主成分置为0
    img_kl_block_vec[:, principal_n:] = 0
    return np.dot(img_kl_block_vec, fvec.T)


blocksize = 2   # 像素块
principal_n1 = 16   # 贡献度高的前N个特征值个数
# for img_name in os.listdir(file_dir):
#     img_path = file_dir + img_name
#     img = cv2.imread(img_path)
#     image_w, image_h = img.shape
#     new_w, new_h = image_w // blocksize * blocksize, image_h // blocksize * blocksize
#     img = cv2.resize(img, (new_h, new_w))
#     vec = img2vec(img, blocksize)
#     kl1 = vec2img(kl_transform(vec, principal_n1), blocksize)
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(img, 'gray')
#     plt.subplot(1, 2, 2)
#     plt.imshow(kl1, 'gray')

# 展示单张图片的K-L结果
# img = cv2.imread("./face/face/glasses/1290.jpg")
# image_w, image_h = 128, 128
# new_w, new_h = image_w // blocksize * blocksize, image_h // blocksize * blocksize
# img = cv2.resize(img, (new_h, new_w))
# vec = img2vec(img, blocksize)
# kl1 = vec2img(kl_transform(vec, principal_n1), blocksize)
# print("finish")
# plt.subplot(1, 2, 1)
# plt.imshow(img, 'gray')
# plt.subplot(1, 2, 2)
# plt.imshow(kl1, 'gray')
# plt.show()


file_dir = "./face/face/glasses/"
svc = SVC()   # 使用sklearn中的支持向量机模型
X = []   # 存放图像名称
Y = []   # 存放分类名称
for f in os.listdir(file_dir):   # 从眼镜类中提取图片
    img = cv2.imread(file_dir + f)
    # 进行K-L变换
    image_w, image_h = 128, 128
    new_w, new_h = image_w // blocksize * blocksize, image_h // blocksize * blocksize
    img = cv2.resize(img, (new_h, new_w))
    X.append(img)
    Y.append("glasses")

file_dir = "./face/face/non_glasses/"
for f in os.listdir(file_dir):   # 从非眼镜类中提取图片
    img = cv2.imread(file_dir + f)
    # 进行K-L变换
    image_w, image_h = 128, 128
    new_w, new_h = image_w // blocksize * blocksize, image_h // blocksize * blocksize
    img = cv2.resize(img, (new_h, new_w))
    X.append(img)
    Y.append("non_glasses")

# 转为numpy数组便于后续运算
X = np.array(X)
Y = np.array(Y)

# 用train_test_split()函数将数据分割为训练集和测试集。test_size=0.2表示测试集占总数据的20%，random_state=0用于确保每次分割都能得到相同的结果
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# 对数组进行变换以便模型计算
X_train = X_train.reshape(X_train.shape[0], 128*128*3)
X_test = X_test.reshape(X_test.shape[0], 128*128*3)

svc.fit(X_train, Y_train)   # 训练模型
Y_dtc = svc.predict(X_test)   # 模型预测
# 输出预测得分并展示结果
acc = accuracy_score(Y_test, Y_dtc)
precision = precision_score(Y_test, Y_dtc, pos_label="glasses")
recall = recall_score(Y_test, Y_dtc, pos_label="glasses")
f1 = f1_score(Y_test, Y_dtc, pos_label="glasses")
print("acc:", acc)
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)
print("Y_dtc:", Y_dtc)
print("Y_test", Y_test)
