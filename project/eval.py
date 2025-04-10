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


listeshoplifting=[[1550  ,2000    ],
[2200  ,4900    ],
[720  ,930    ] ,
[2010  ,2160    ],
[630  ,720    ],
[360  ,420    ],
[2340  ,2460    ],
[2070  ,2220    ],
[570  ,840    ],
[1020  ,1470    ],
[120  ,330    ],
[630  ,750    ],
[7350  ,7470    ],
[1140  ,1200    ],
[2190  ,2340    ],
[11070 , 11250  ] ,
[1020  ,1350   ]]
def creatliste(numframes,listeshop):

    pas= numframes // 32
    liste = np.zeros(32)
    for i in range(32):
        ii=i*pas

        if  ii >= listeshop[0] and ii <= listeshop[1] :
            liste[i]=1

    return liste

def accu(liste , predections2):
    a=0
    for i in range(32):
        if (liste[i] == predections2[i]):
            a=a+1

    acc= (a / 32 )*100
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


def run_demo():
    print("ana howa")
    accgraphe=[]
    with open('E:\\Anomaly-Detection-Dataset\\Ntest.txt') as f:
        Nlines = [line.rstrip('\n') for line in f]
    with open('E:\\Anomaly-Detection-Dataset\\Shoptest.txt') as f:
        shoplines = [line.rstrip('\n') for line in f]
    with open('E:\\Anomaly-Detection-Dataset\\test.txt') as f:
        vdlines = [line.rstrip('\n') for line in f]



    # build models
    classifier_model = build_classifier_model()

    print("model saved ")
    print("Models initialized")
    accliste=[]

    for jj in range(len(vdlines)):

        video_name = os.path.basename(vdlines[jj]).split('.')[0]

        # read video



        video_clips, num_frames = get_video_clips(vdlines[jj])


        if (str(video_name) in Nlines):
            rgb_feature_bag = np.loadtxt("E:\\Anomaly-Detection-Dataset\\newdata\\normaltest\\" + video_name + ".txt")
        if (str(video_name) in shoplines):
            rgb_feature_bag = np.loadtxt("E:\\Anomaly-Detection-Dataset\\newdata\\shoplifting\\" + video_name + ".txt")

        print(rgb_feature_bag.shape)
        # classify using the trained classifier model
        predictions = classifier_model.predict(rgb_feature_bag)


        predictions2=[]
        for iii in range(len(predictions)):
            if(predictions[iii]<0.5 ):
                predictions2.append(0)
            else:
                predictions2.append(1)

        if (str(video_name) in Nlines):

            liste =  np.zeros(32)
        if (str(video_name) in shoplines):
            l=listeshoplifting[jj]
            liste = creatliste(num_frames,l)


        acc = accu(liste,predictions2)
        print("accu",acc)
        accliste.append(acc)
        with open('D:\\resultats\\accuracy1.txt', 'a', encoding='utf-8') as my_file:
            my_file.write(str(video_name) + " "+ str(acc) + " %" + '\n')


        print('Executed Successfully - ' + video_name )
        moyenacc=sum(accliste) / len(accliste)
        print("moyenne ",moyenacc)
        accgraphe = np.hstack((accgraphe, moyenacc))

    plt.plot(accgraphe)
    plt.title('Test accuracy')
    plt.xlabel('videos')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    run_demo()