from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adagrad
from scipy.io import savemat
from keras.models import model_from_json
import tensorflow as tf
import os
from os import listdir
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def save_model(model, json_path, weight_path):
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)


def load_model(json_path):
    model = model_from_json(open(json_path).read())
    return model


def load_dataset_Train_batch(AbnormalPath, NormalPath):
    batchsize = 60
    n_exp = int(batchsize / 2)
    Num_abnormal = 70
    Num_Normal = 219
    Abnor_list_iter = np.random.permutation(Num_abnormal)
    Abnor_list_iter = Abnor_list_iter[Num_abnormal - n_exp:]
    Norm_list_iter = np.random.permutation(Num_Normal)
    Norm_list_iter = Norm_list_iter[Num_Normal - n_exp:]
    All_Videos = []
    with open(AbnormalPath + "Shoplifting.txt", 'r') as f1:  # file contain path to anomaly video file.
        for line in f1:
            All_Videos.append(line.strip())
    AllFeatures = []
    Video_count = -1
    for iv in Abnor_list_iter:
        Video_count = Video_count + 1
        VideoPath = os.path.join(AbnormalPath, All_Videos[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        num_feat = len(words) / 4096
        count = -1;
        VideoFeatues = []
        for feat in range(0, int(num_feat)):
            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                VideoFeatues = feat_row1
            if count > 0:
                VideoFeatues = np.vstack((VideoFeatues, feat_row1))
        if Video_count == 0:
            AllFeatures = VideoFeatues
        if Video_count > 0:
            AllFeatures = np.vstack((AllFeatures, VideoFeatues))
    All_Videos = []
    with open(NormalPath + "Normal.txt", 'r') as f1:  # file contain path to normal video file.
        for line in f1:
            All_Videos.append(line.strip())
    for iv in Norm_list_iter:
        VideoPath = os.path.join(NormalPath, All_Videos[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        feat_row1 = np.array([])
        num_feat = len(words) / 4096
        count = -1;
        VideoFeatues = []
        for feat in range(0, int(num_feat)):
            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                VideoFeatues = feat_row1
            if count > 0:
                VideoFeatues = np.vstack((VideoFeatues, feat_row1))
            feat_row1 = []
        AllFeatures = np.vstack((AllFeatures, VideoFeatues))
    AllLabels = np.zeros(32 * batchsize, dtype='uint8')
    th_loop1 = n_exp * 32
    th_loop2 = n_exp * 32 - 1
    for iv in range(0, 32 * batchsize):
        if iv < th_loop1:
            AllLabels[iv] = int(0)
        if iv > th_loop2:
            AllLabels[iv] = int(1)
    return AllFeatures, AllLabels












def custom_objective(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    n_seg = 32
    nvid = 60
    n_exp = nvid // 2
    Num_d = 32*nvid
    sub_max = tf.ones_like(y_pred)
    sub_sum_l1 = tf.ones_like(y_true)
    sub_l2 = tf.ones_like(y_true)
    sub_sum_labels = tf.ones_like(y_true)
    for ii in range(0, nvid, 1):
        mm = y_true[ii * n_seg:ii * n_seg + n_seg]
        sub_sum_labels = tf.concat([sub_sum_labels, tf.reduce_sum(mm, axis=0, keepdims=True)], axis=0)
        Feat_Score = y_pred[ii * n_seg:ii * n_seg + n_seg]
        sub_max = tf.concat([sub_max, tf.expand_dims(tf.reduce_max(Feat_Score), axis=0)], axis=0)
        sub_sum_l1 = tf.concat([sub_sum_l1, tf.expand_dims(tf.cast(tf.reduce_sum(tf.cast(Feat_Score, tf.float32)), tf.uint8), axis=0)],axis=0)
        z1 = tf.ones_like(Feat_Score)
        z2 = tf.concat([z1, Feat_Score], axis=0)
        z3 = tf.concat([Feat_Score, z1], axis=0)
        z_22 = z2[31:]
        z_44 = z3[:33]
        z = z_22 - z_44
        z = z[1:32]
        z = tf.reduce_sum(tf.square(z))
        sub_l2 = tf.concat([sub_l2, tf.expand_dims(tf.cast(tf.stack(tf.cast(z, tf.uint8)), tf.uint8), axis=0)], axis=0)
    sub_score = sub_max[Num_d:]
    F_labels = sub_sum_labels[Num_d:]
    sub_sum_l1 = sub_sum_l1[Num_d:]
    sub_sum_l1 = sub_sum_l1[:n_exp]
    sub_l2 = sub_l2[Num_d:]
    sub_l2 = sub_l2[:n_exp]
    indx_nor = tf.where(tf.equal(F_labels, 32))[:, 0]
    indx_abn = tf.where(tf.equal(F_labels, 0))[:, 0]
    n_Nor = n_exp
    Sub_Nor = tf.gather(sub_score, indx_nor)
    Sub_Abn = tf.gather(sub_score, indx_abn)
    z = tf.ones_like(y_true)
    for ii in range(0, n_Nor, 1):
        sub_z = tf.maximum(1 - Sub_Abn + Sub_Nor[ii], 0)
        z = tf.concat([tf.cast(z, tf.float32), tf.expand_dims(tf.reduce_sum(sub_z), 0)], axis=0)
    z = z[Num_d:]
    z = tf.reduce_mean(z, axis=-1) + 0.00008*tf.reduce_sum(tf.cast(sub_sum_l1, tf.float32)) + 0.00008*tf.reduce_sum(tf.cast(sub_l2, tf.float32))
    return z



# Path contains C3D features (.txt file) of each video.
# Each text file contains 32 features, each of 4096 dimension
AllClassPath = 'D:\\newdatari\\'

output_dir = 'D:\\newdatari\\'

# Output_dir save trained weights and model.

weights_path = output_dir + 'myweights5.mat'

model_path = output_dir + 'mymodel5.json'

# Create Full connected Model
model = Sequential()

model.add(Dense(512, input_dim=4096,kernel_initializer='glorot_normal',kernel_regularizer=l2(0.001),activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(32,kernel_initializer='glorot_normal',kernel_regularizer=l2(0.001)))
model.add(Dropout(0.6))
model.add(Dense(1,kernel_initializer='glorot_normal',kernel_regularizer=l2(0.001),activation='sigmoid'))
adagrad=Adagrad(lr=0.01, epsilon=1e-08)
model.compile(loss=custom_objective, optimizer=adagrad, metrics=['accuracy'])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

All_class_files = listdir(AllClassPath)
All_class_files.sort()
loss_graph = []
accuracy_list = []
num_iters = 2000
total_iterations = 0
batchsize = 60
time_before = datetime.now()
# Define the number of iterations between accuracy evaluations
eval_freq = 100
for it_num in range(num_iters):
    inputs, targets = load_dataset_Train_batch(AllClassPath, AllClassPath)
    batch_loss = model.train_on_batch(inputs, targets)
    loss_graph = np.append(loss_graph, batch_loss)
    total_iterations += 1
    if total_iterations % 20 == 1:
        print("Iteration=" + str(total_iterations) + " took: " + str(
            datetime.now() - time_before) + ", with loss of " + str(batch_loss))

print(len(loss_graph))
plt.plot(loss_graph)
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()


print("Train Successful - Model saved")
tf.keras.models.save_model(model, 'myoptimazedmodel2')
save_model(model, model_path, weights_path)
model.save('myoptimazedmodel.h5')