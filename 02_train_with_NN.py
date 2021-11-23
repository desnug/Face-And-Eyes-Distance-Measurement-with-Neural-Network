import cv2
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Menampilkan dataset
df = pd.read_csv(r'dataset_pixel.csv', sep=',', header=None)
print(df)

# Membuat Neural Network
# definisi fungsi fungsi


def normalize(data, minimal, maximal):
    y = np.float32(data)
    for x in range(data.size):
        y[x] = 2*(data[x]-minimal)/(maximal-minimal) - 1
    return y


def denormalize(data, minimal, maximal):
    y = np.float32(data)
    for x in range(data.size):
        y[x] = 0.5*(data[x]+1)*(maximal-minimal) + minimal
    return y


def relu(x):
    y = np.maximum(0.0, x)
    return y


def drelu(x):
    y = np.where(x > 0, 1, 0)
    return y


def line(x):
    y = x
    return y


def dline(x):
    y = 1
    return y


def tansig(x):
    y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    return y


def dtansig(x):
    y = 1 - ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))**2
    return y


def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y


def dsigmoid(x):
    y = (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))
    return y


# Mengambil input dan target dari dataset
dm = np.asmatrix(df.T)
input_data = np.asarray(dm[0]).reshape(-1)
output_data = np.asarray(dm[4]).reshape(-1)

# Normalisasi data
i_data = normalize(input_data, 0, 480)
o_data = normalize(output_data, 0, 480)

# Generate random w and b
w1 = np.random.uniform(-1, 1, (10, 1))
b1 = np.random.uniform(-1, 1, (10, 1))
w2 = np.random.uniform(-1, 1, (1, 10))
b2 = np.random.uniform(-1, 1, (1, 1))

# forward and backward
epoch = 0
epoch_max = 500
loss_max = 1e-6
loss_total = 1
a = 0.1  # Learning rate
while(loss_total > loss_max and epoch < epoch_max):
    # for z in range(epoch_max):
    epoch = epoch + 1
    loss_total = 0
    i_data = np.roll(i_data, epoch)
    o_data = np.roll(o_data, epoch)
    for x in range(i_data.size):
        h_layer = sigmoid(
            np.add(np.matmul(w1, i_data[x].reshape(1, 1)), b1))  # F1(w1*x+b1)
        o_layer = tansig(np.add(np.matmul(w2, h_layer), b2))  # F2(w2*x+b2)

        loss = 0.5*(o_data[x]-o_layer)**2
        if loss > loss_max:

            s0 = -2*np.multiply(dtansig(o_layer), (o_data[x]-o_layer))
            s1 = np.multiply(dsigmoid(h_layer), w2.T)*s0

            w2 = w2-a*(np.multiply(s0, h_layer.T))
            b2 = b2-a*s0

            w1 = w1-a*(np.multiply(s1, i_data[x]))
            b1 = b1-a*s1
        loss_total = loss_total + loss
    loss_total = 1/(2*i_data.size)*loss_total
    print(loss_total)

# Test NN
in_tes = np.arange(0, 480, 1)
out_tes = np.zeros(in_tes.size)
for j in range(in_tes.size):
    tes = sigmoid(np.add(np.matmul(
        w1, (normalize(in_tes[j].reshape(1, 1), 0, 480))), b1))  # F1(w1*x+b1)
    tes1 = tansig(np.add(np.matmul(w2, tes), b2))  # F2(w2*x+b2)
    out_tes[j] = denormalize(tes1, 0, 480)
title = 'Grafik Perbandingan Jarak Wajah Ke Kamera\n'
subtitle = 'Jarak Prediksi vs Jarak Real'
plt.title(title + subtitle)
plt.plot(input_data, output_data, 'co', label="perhitungan manual")
plt.plot(in_tes, out_tes, 'r--', label="prediksi")
plt.legend(loc="upper right")
plt.xlabel('input (data piksel wajah)')
plt.ylabel('output (jarak)')
plt.show()

# Save w dan b
np.save('weightbias/w1.npy', w1)
np.save('weightbias/b1.npy', b1)
np.save('weightbias/w2.npy', w2)
np.save('weightbias/b2.npy', b2)

# show numpy file
print('Print Numpy File: ')
data_w1 = np.load('weightbias/w1.npy')
data_b1 = np.load('weightbias/b1.npy')
data_w2 = np.load('weightbias/w2.npy')
data_b2 = np.load('weightbias/b2.npy')
print(data_w1)
# print(data_b1)
# print(data_w2)
# print(data_b2)
