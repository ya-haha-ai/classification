from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
import rasterio as rio
import earthpy.plot as ep
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import *
import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
# 数据读取
S_sentinel_bands = glob("E:/3d-data/sundarbans_data-main/*B?*.tiff")
S_sentinel_bands.sort()

l = []
for i in S_sentinel_bands:
    with rio.open(i, 'r') as f:
        l.append(f.read(1))

# Data
arr_st = np.stack(l)

# Ground Truth
y_data = loadmat('E:/3d-data/sundarbans_data-main/Sundarbands_gt.mat')
print(y_data)

# # 数据可视化
# #   真实地面
# ep.plot_rgb(
#     arr_st,
#     rgb=(3, 2, 1),
#     stretch=True,
#     str_clip=0.02,
#     figsize=(12, 16),
#     # title="RGB Composite Image with Stretch Applied",
# )
#
# plt.show()
# #   合成图像
# ep.plot_bands(y_data,
#               cmap=matplotlib.colors.ListedColormap(['darkgreen', 'green', 'black',
#                                    '#CA6F1E', 'navy', 'forestgreen']))
# plt.show()
# #   12个波段图像
# ep.plot_bands(arr_st,
#               cmap = 'gist_earth',
#               figsize = (20, 12),
#               cols = 6,
#               cbar = False)
# plt.show()

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=False):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test


dataset = 'SB'
test_size = 0.30
windowSize = 15
MODEL_NAME = 'Sundarbans'
path = 'E:/3d-data/'

X_data = np.moveaxis(arr_st, 0, -1)
y_data = loadmat('E:/3d-data/sundarbans_data-main/Sundarbands_gt.mat')['gt']

# Apply PCA
K = 5
X, pca = applyPCA(X_data, numComponents=K)

print(f'Data After PCA: {X.shape}')

# Create 3D Patches
X, y = createImageCubes(X, y_data, windowSize=windowSize)
print(f'Patch size: {X.shape}')

# Split train and test
X_train, X_test, y_train, y_test = splitTrainTestSet(X, y, testRatio=test_size)

X_train = X_train.reshape(-1, windowSize, windowSize, K, 1)
X_test = X_test.reshape(-1, windowSize, windowSize, K, 1)

# One Hot Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(f'Train: {X_train.shape}\nTest: {X_test.shape}\nTrain Labels: {y_train.shape}\nTest Labels: {y_test.shape}')
S = windowSize
L = K
output_units = y_train.shape[1]

## input layer
input_layer = tf.keras.layers.Input((S, S, L, 1))

## convolutional layers
conv_layer1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(2, 2, 3), activation='relu')(input_layer)
conv_layer2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(2, 2, 3), activation='relu')(conv_layer1)
conv2d_shape = conv_layer2.shape
conv_layer3 = tf.keras.layers.Reshape((conv2d_shape[1], conv2d_shape[2], conv2d_shape[3] * conv2d_shape[4]))(
    conv_layer2)
conv_layer4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu')(conv_layer3)

flatten_layer = tf.keras.layers.Flatten()(conv_layer4)

## fully connected layers
dense_layer1 = tf.keras.layers.Dense(128, activation='relu')(flatten_layer)
dense_layer1 = tf.keras.layers.Dropout(0.4)(dense_layer1)
dense_layer2 = tf.keras.layers.Dense(64, activation='relu')(dense_layer1)
dense_layer2 = tf.keras.layers.Dropout(0.4)(dense_layer2)
dense_layer3 = tf.keras.layers.Dense(20, activation='relu')(dense_layer2)
dense_layer3 = tf.keras.layers.Dropout(0.4)(dense_layer3)
output_layer = tf.keras.layers.Dense(units=output_units, activation='softmax')(dense_layer3)
# define the model with input layer and output layer
model = tf.keras.Model(name=dataset + '_Model', inputs=input_layer, outputs=output_layer)

model.summary()

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
logdir = path + "logs/" + model.name + '_' + datetime.now().strftime("%Y%m%d-%H_%M_%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=1,
                                      verbose=1,
                                      restore_best_weights=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='Pavia_University_Model.h5',
                                                monitor='val_loss',
                                                mode='min',
                                                save_best_only=True,
                                                verbose=1)
# Fit
history = model.fit(x=X_train, y=y_train,
                    batch_size=1024*6, epochs=6,
                    validation_data=(X_test, y_test), callbacks=[tensorboard_callback, es, checkpoint])


history = pd.DataFrame(history.history)

plt.figure(figsize=(12, 6))
plt.plot(range(len(history['accuracy'].values.tolist())), history['accuracy'].values.tolist(), label='Train_Accuracy')
plt.plot(range(len(history['loss'].values.tolist())), history['loss'].values.tolist(), label='Train_Loss')
plt.plot(range(len(history['val_accuracy'].values.tolist())), history['val_accuracy'].values.tolist(),
         label='Test_Accuracy')
plt.plot(range(len(history['val_loss'].values.tolist())), history['val_loss'].values.tolist(), label='Test_Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()

pred = model.predict(X_test, batch_size=1204 * 6, verbose=1)

plt.figure(figsize=(10, 7))

classes = [f'Class-{i}' for i in range(1, 7)]

mat = confusion_matrix(np.argmax(y_test, 1),
                       np.argmax(pred, 1))

df_cm = pd.DataFrame(mat, index=classes, columns=classes)

print(classification_report(np.argmax(y_test, 1),
                            np.argmax(pred, 1),
      target_names = [f'Class-{i}' for i in range(1, 7)]))

sns.heatmap(df_cm, annot=True, fmt='d')

plt.show()

pred_t = model.predict(X.reshape(-1, windowSize, windowSize, K, 1),
                       batch_size=1204 * 6, verbose=1)
# Visualize Groundtruth

ep.plot_bands(np.argmax(pred_t, axis=1).reshape(954, 298),
              cmap=ListedColormap(['darkgreen', 'green', 'black',
                                   '#CA6F1E', 'navy', 'forestgreen']))
plt.show()

