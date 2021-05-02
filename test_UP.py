# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
from keras.regularizers import l2
import time
import collections
from sklearn import metrics, preprocessing
from Utils import zeroPadding, normalization, modelStatsRecord, averageAccuracy, Dense_DFFN_IN, residual_DFFN
import os
import keras
import cv2
from keras.losses import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 产生新数据集的过程
# indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
# 训练集 ，151 ，151， 3
def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        # counter 是从0开始计数的，是具体的值
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch

# divide dataset into train and test datasets
def sampling(proptionVal, groundTruth):
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    print(m)
    # 16
    # 16类，对每一类样本要先打乱，然后再按比例分配，得到一个字典，因为上面是枚举，所以样本和标签的对应
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        # print(indices)
        # 每一类的样本数
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    # 将所有的训练样本存到train集合中，将所有的测试样本存到test集合中
    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    print(len(test_indices))
    # 8194
    print(len(train_indices))
    # 2055
    return train_indices, test_indices

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

# 调用设计好的模型
def model_DFFN():
    model_dense, eval_model = Dense_DFFN_IN.ResnetBuilder.build_resnet_8((1, img_rows, img_cols, img_channels),
                                                                nb_classes)

    RMS = RMSprop(lr=0.0003)
    # Let's train the model using RMSprop
    # model_dense.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])
    model_dense.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
                        loss_weights=[1., 0.5],
                        optimizer=RMS,
                        metrics=['accuracy'])

    return model_dense, eval_model

# 加载数据
# 修正的Indian pines数据集
mat_data = sio.loadmat('./datasets/UP/PaviaU.mat')
data_IN = mat_data['paviaU']
# 标签数据
mat_gt = sio.loadmat('./datasets/UP/PaviaU_gt.mat')
gt_IN = mat_gt['paviaU_gt']

print(data_IN.shape)
# (145,145,200)
print(gt_IN.shape)
# (145,145)

# 中值滤波
# data_IN = cv2.medianBlur(data_IN, 5)

# 高斯滤波
# data_IN = cv2.GaussianBlur(data_IN, (5, 5), 0)

# 均值滤波
# data_IN = cv2.blur(data_IN, (5, 5))

# new_gt_IN = set_zeros(gt_IN, [1,4,7,9,13,15,16])
new_gt_IN = gt_IN

batch_size = 16
nb_classes = 9
nb_epoch = 200  # 400
img_rows, img_cols = 17, 17  # 27, 27
patience = 200

INPUT_DIMENSION_CONV = 103


INPUT_DIMENSION = 103

# 20%:10%:70% data for training, validation and testing

TOTAL_SIZE = 42776
VAL_SIZE = 4281

TRAIN_SIZE = 4281
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.9  # 20% for trainnig and 80% for validation and testing


save_dir = './result/models-up-dfdn-17-3-10%'

img_channels = 103
PATCH_LENGTH = 8  # Patch_size (13*2+1)*(13*2+1)

print(data_IN.shape[:2])
# (145,145)
print(np.prod(data_IN.shape[:2]))
# 21025
print(data_IN.shape[2:])
# (200,)
print(np.prod(data_IN.shape[2:]))
# 200
print(np.prod(new_gt_IN.shape[:2]))
# 21025

# 对数据进行reshape处理之后，进行scale操作
data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

# 标准化操作，即将所有数据沿行沿列均归一化道0-1之间
data = preprocessing.scale(data)
print(data.shape)
# (21025, 200)

# scaler = preprocessing.MaxAbsScaler()
# data = scaler.fit_transform(data)

# 对数据边缘进行填充操作，有点类似之前的镜像操作
data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
whole_data = data_
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)
print(padded_data.shape)
# (151, 151, 200)
# 因为选择的是7*7的滑动窗口，145*145,145/7余5，也就是说有5个像素点扫描不到，所有在长宽每边个填充3，也就是6，这样的话
# 就可以将所有像素点扫描到

ITER = 1
CATEGORY = 9

train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
print(train_data.shape)
# (2055, 7, 7, 200)
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
print(test_data.shape)
# (8194, 7, 7, 200)

# 评价指标
KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CATEGORY))

# seeds = [1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229]

seeds = [1334, 1335, 1336]

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))
    # # 1 Iteration

    # save the best validated model
    # 使用easystopping通过一个动态阈值去选择最优的模型
    best_weights_DFFN_path = save_dir + '/UP_best_3D_DFFN_0505' + str(
        index_iter + 1) + ' .hdf5'

    # 通过sampling函数拿到测试和训练样本
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    # train_indices 2055     test_indices 8094

    # gt本身是标签类，从标签类中取出相应的标签 -1，转成one-hot形式
    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    # 这个地方论文也解释了一下，是新建了一个以采集中心为主的新数据集，还是对元数据集进行了一些更改
    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    # 拿到了新的数据集进行reshpae之后，数据处理就结束了
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

    # 在测试数据集上进行验证和测试的划分
    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    gt_test = gt[test_indices] - 1
    gt_test = gt_test[:-VAL_SIZE]

    model_dense, eval_model = model_DFFN()
    model_dense.load_weights(best_weights_DFFN_path)

    tic7 = time.clock()
    pred_test = eval_model.predict(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)
    collections.Counter(pred_test)
    toc7 = time.clock()
    collections.Counter(pred_test)

    print('Test time:', toc7 - tic7)

    # print(len(gt_test))
    # 8194
    # 这是测试集，验证和测试还没有分开
    overall_acc = metrics.accuracy_score(pred_test, gt_test)
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test)
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test)
    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    # TRAINING_TIME_3D_SEN.append(toc6 - tic6)
    TESTING_TIME.append(toc7 - tic7)
    ELEMENT_ACC[index_iter, :] = each_acc

    print("Test finished.")
    print("# %d Iteration" % (index_iter + 1))

# 自定义输出类
modelStatsRecord.outputStats_assess_(KAPPA, OA, AA, ELEMENT_ACC, TESTING_TIME, CATEGORY,
                                     save_dir + '/KSC_test.txt',
                                     save_dir + '/KSC_test_element.txt')