import os
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from c3d import *
from classifier import *
import matplotlib
matplotlib.use('TkAgg')
import cv2
import numpy as np
import time
from keras.utils.data_utils import get_file

C3D_MEAN_PATH = 'D:\\openvinomymodel\\infernce\\c3d_mean.npy'


def preprocess_input(video):
    intervals = np.ceil(np.linspace(0, video.shape[0] - 1, 16)).astype(int)
    frames = video[intervals]

    # Reshape to 128x171
    reshape_frames = np.zeros((frames.shape[0], 128, 171, frames.shape[3]))
    for i, img in enumerate(frames):
        img = cv2.resize(src=img,
                         dsize=(171, 128),
                         interpolation=cv2.INTER_CUBIC)
        reshape_frames[i, :, :, :] = img

    mean_path = get_file('c3d_mean.npy',
                         C3D_MEAN_PATH,
                         cache_subdir='models',
                         md5_hash='08a07d9761e76097985124d9e8b2fe34')

    mean = np.load(mean_path)
    reshape_frames -= mean
    # Crop to 112x112
    reshape_frames = reshape_frames[:, 8:120, 30:142, :]
    # Add extra dimension for samples
    reshape_frames = np.expand_dims(reshape_frames, axis=0)

    return reshape_frames


def interpolate(features, features_per_bag):
    feature_size = np.array(features).shape[1]
    interpolated_features = np.zeros((features_per_bag, feature_size))
    interpolation_indicies = np.round(np.linspace(0, len(features) - 1, num=features_per_bag + 1))
    count = 0
    for index in range(0, len(interpolation_indicies) - 1):
        start = int(interpolation_indicies[index])
        end = int(interpolation_indicies[index + 1])

        assert end >= start

        if start == end:
            temp_vect = features[start, :]
        else:
            temp_vect = np.mean(features[start:end + 1, :], axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)

        if np.linalg.norm(temp_vect) == 0:
            print("Error")

        interpolated_features[count, :] = temp_vect
        count = count + 1

    return np.array(interpolated_features)


def extrapolate(outputs, num_frames):
    extrapolated_outputs = []
    extrapolation_indicies = np.round(np.linspace(0, len(outputs) - 1, num=num_frames))
    for index in extrapolation_indicies:
        extrapolated_outputs.append(outputs[int(index)])
    return np.array(extrapolated_outputs)


frame_height = 240
frame_width = 320
channels = 3
frame_count = 16
features_per_bag = 32


def sliding_window(arr, size, stride):
    num_chunks = int((len(arr) - size) / stride) + 2
    result = []

    for i in range(0, num_chunks * stride, stride):

        if len(arr[i:i + size]) > 0:
            result.append(arr[i:i + size])

    return np.array(result, dtype=object)


def creatliste(numframes, listeshop):
    pas = numframes // 32
    liste = np.zeros(32)
    for i in range(32):
        ii = i * pas

        if ii >= listeshop[0] and ii <= listeshop[1]:
            liste[i] = 1

    return liste


def accu(liste, predections2):
    a = 0
    for i in range(32):
        if (liste[i] == predections2[i]):
            a = a + 1

    acc = (a / 32) * 100
    return acc


def get_video_clips(video_path):
    frames = get_video_frames(video_path)

    clips = sliding_window(frames, 16, 16)
    return clips, len(frames)


def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    return frames


with open('D:\\newdatari\\test.txt') as f:
    vdlines = [line.rstrip('\n') for line in f]

# build models
feature_extractor = c3d_feature_extractor()
classifier_model = build_classifier_model()


model1_times = []
model2_times = []
print("model saved ")
print("Models initialized")
accliste = []

for jj in range(len(vdlines)):

    video_name = os.path.basename(vdlines[jj]).split('.')[0]

    # read video

    video_clips, num_frames = get_video_clips(vdlines[jj])

    # extract features





    print("nombre de clip est ", len(video_clips))
    rgb_features = []
    start = time.time()
    for i, clip in enumerate(video_clips):
        clip = np.array(clip)
        if len(clip) < frame_count:
            continue

        clip = preprocess_input(clip)

        # Infer using the C3D model
        rgb_feature = feature_extractor.predict(clip)[0]
        rgb_features.append(rgb_feature)





        print("Processed clip : ", i)

    end = time.time()
    model1_times.append(end - start)
    t1 = end - start
    rgb_features = np.array(rgb_features)



    # bag features
    rgb_feature_bag = interpolate(rgb_features, features_per_bag)



    # classify using the trained classifier model


    start = time.time()

    predictions = classifier_model.predict(rgb_feature_bag)
    end = time.time()
    model2_times.append(end - start)
    t2 = end - start
    with open('D:\\resultats\\Timeofnormalmodel.txt', 'a', encoding='utf-8') as my_file:
        my_file.write(
            str(video_name) + " model c3d :" + str(t1) + " sc" + " classfication model  :" + str(t2) + " sc" + '\n')

    # Process the output data as needed


print("*********************")
plt.plot(range(len(vdlines)), model1_times, label='Model C3d')
plt.plot(range(len(vdlines)), model2_times, label='Classification Model ')
plt.xlabel('Videos')
plt.ylabel('Inference time (s)')
plt.legend()
plt.show()

