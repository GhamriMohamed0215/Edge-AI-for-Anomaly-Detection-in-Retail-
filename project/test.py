from c3d import *
from classifier import *
from visualization_util import *
from array_util import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np

frame_height = 240
frame_width = 320
channels = 3
frame_count = 16
features_per_bag = 32

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


def run_demo():

    video_name = os.path.basename("D:\\input\\myShoplifting01.mp4").split('.')[0]

    # read video
    video_clips, num_frames = get_video_clips("D:\\input\\myShoplifting01.mp4")

    print("Number of clips in the video : ", len(video_clips))

    # build models
    feature_extractor = c3d_feature_extractor()
    classifier_model = build_classifier_model()

    print("Models initialized")

    # extract features
    rgb_features = []
    for i, clip in enumerate(video_clips):
        clip = np.array(clip)
        if len(clip) < frame_count:
            continue

        clip = preprocess_input(clip)
        rgb_feature = feature_extractor.predict(clip)[0]
        rgb_features.append(rgb_feature)

        print("Processed clip : ", i)

    rgb_features = np.array(rgb_features)

    # bag features
    rgb_feature_bag = interpolate(rgb_features, features_per_bag)


    np.savetxt("D:\\output\\vd6.txt",rgb_feature_bag)

    # classify using the trained classifier model
    predictions = classifier_model.predict(rgb_feature_bag)

    print("prediction1")
    print(predictions)

    predictions = np.array(predictions).squeeze()

    predictions = extrapolate(predictions, num_frames)

    save_path = os.path.join("D:\\output\\", video_name + '.gif')
    # visualize predictions
    print('Executed Successfully - ' + video_name + '.gif saved')



if __name__ == '__main__':
    run_demo()