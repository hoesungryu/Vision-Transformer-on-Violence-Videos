# import
import os
from pathlib import Path
import json
import re
from tqdm import tqdm
import cv2
from torchvision import transforms
import numpy as np
import glob
import argparse
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split

# args
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=False,
                    default='efficient',help="efficient or resnet3d")
args = parser.parse_args()


# user_defined function 
def name2label(name):
    if bool(re.search('NV_*', name) == True):
        y = 0 
    else: 
        y = 1
    return y 

def move_dir(X,dst):
    for video in X:
        save_path = os.path.join(dst,os.path.basename(video))
        copyfile(video, save_path)

class Preprocessing:
    """
    Args:
    file_dir: source folder of target videos
    save_dir: destination folder of output .npy files
    """

    def __init__(self, file_path, save_path, mode,resize=(224, 224)):
        self.resize = resize
        self.file_path = file_path
        self.save_dir = save_path

    def video2npy(self, video):
        cap = cv2.VideoCapture(video)  # load video
        len_frames = int(cap.get(7))  # get number of frames

        try:
            frames = []
            for i in range(len_frames-1):
                _, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transforms.ToTensor()(frame)
                frame = transforms.Resize(self.resize)(frame)

                if self.mode == 'efficient':
                # efficient-Net
#                 frame = transforms.Normalize([0.485, 0.456, 0.406],
#                                              [0.229, 0.224, 0.225])(frame)
                elif self.mode == 'resnet3d':
                frame = transforms.Normalize([0.43216, 0.394666, 0.37645],
                                             [0.22803, 0.22145, 0.216989])(frame)
                
                frame = np.float32(frame.numpy())
                frames.append(frame)
        except:
            print('Error: ', video, len_frames)

        finally:
            frames = np.array(frames)
            cap.release()

        return frames

    def save2npy(self):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        videos = glob.glob(self.file_path+'/*')  # List the files
#         videos = [x for x in videos if x.endswith('.mp4')]

        for video in tqdm(videos):
            video_name = video.split('/')[-1].split('.')[0]  # Split video name

            # Get dest
            save_path = os.path.join(self.save_dir, video_name+'.npy')

            # Load and preprocess video
            data = self.video2npy(video)

            # Save as .npy file
            np.save(save_path, data)

        return None


def getOpticalFlow(video):
    """Calculate dense optical flow of input video
    Args:
        video: the input video with shape of [frames,height,width,channel]. dtype=np.array
    Returns:
        flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
        flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
    """
    # initialize the list of optical flows
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img,(224,224,1)))

    flows = []
    for i in range(0,len(video)-1):
        # calculate optical flow between each pair of frames
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # subtract the mean in order to eliminate the movement of camera
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
        # Add into list 
        flows.append(flow)
        
    # Padding the last frame as empty array
    flows.append(np.zeros((224,224,2)))
      
    return np.array(flows, dtype=np.float32)

def Video2Npy(file_path, resize=(224,224)):
    """Load video and tansfer it into .npy format
    Args:
        file_path: the path of video file
        resize: the target resolution of output video
    Returns:
        frames: gray-scale video
        flows: magnitude video of optical flows 
    """
    # Load video
    cap = cv2.VideoCapture(file_path)
    # Get number of frames
    len_frames = int(cap.get(7))
    # Extract frames from video
    try:
        frames = []
        for i in range(len_frames-1):
            _, frame = cap.read()
    
            frame = cv2.resize(frame,resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224,224,3))
            frames.append(frame)   
    except:
        print("Error: ", file_path, len_frames,i)
    finally:
        frames = np.array(frames)
        cap.release()
            
    # Get the optical flow of video
    flows = getOpticalFlow(frames)
    
    result = np.zeros((len(flows),224,224,5))
    result[...,:3] = frames
    result[...,3:] = flows
    
    return result

def Save2Npy(file_dir, save_dir):
    """Transfer all the videos and save them into specified directory
    Args:
        file_dir: source folder of target videos
        save_dir: destination folder of output .npy files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # List the files
    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        # Split video name
        video_name = v.split('.')[0]
        # Get src 
        video_path = os.path.join(file_dir, v)
        # Get dest 
        save_path = os.path.join(save_dir, video_name+'.npy') 
        # Load and preprocess video
        data = Video2Npy(file_path=video_path, resize=(224,224))
#         data = np.uint8(data)
        # Save as .npy file
        np.save(save_path, data)
    
    return None


def traintestsplit_(npy_path):
    file_list = glob.glob(npy_path + '/*')
    y = []
    for file in file_list:
        name = os.path.basename(file)
        label = name2label(name)
        y.append(label)
    X_train, X_test, y_train, y_test = train_test_split(file_list,y,test_size=0.3,random_state=42,stratify=y)
    move_dir(X_train, dst_train)
    move_dir(X_test, dst_test)

    return None 


def main(args):
    mode = args.mode
    source_path = '../data/Real Life Violence Dataset/'
    target_path = f'../data/npy_{mode}/'
    

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for category in ['NonViolence','Violence']:
        path_dir = os.path.join(source_path,category)
        save_dir = os.path.join(target_path)
        
        if mode == '3dfused':
            Save2Npy(file_dir=path_dir, save_dir=save_dir)
            traintestsplit_(save_dir)



        else:
            Preprocessing(path_dir,save_dir, mode,resize=(112,112)).save2npy()
    
    print('[INFO] Done ...')

if __name__ == '__main__':
    main(args)