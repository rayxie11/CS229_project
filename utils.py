'''
This file contains all the util functions that are needed to extract information from dataset
'''
import os
import json
import numpy as np
import cv2


'''
This function converts labels {0,10} into onehot labels
Args: 
    n: An int label
Returns: 
    A numpy array of onehot labels
Example: 
    6 -> [0,0,0,0,0,0,1,0,0,0,0]
'''
def convert_onehot(n):
    zeros = np.zeros(11)
    zeros[n] = 1

    return zeros

'''
This function converts all labels in dataset to onehot labels
Args:
    file_location: A string for the file location of dataset labels
Returns:
    file: A list of file names (n,)
    label: A 2d numpy array of file labels (n,11)
'''
def onehot_labels(file_location):
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.split(cur_dir)[0]
    f = open(parent_dir+'/dataset/'+file_location)
    data = json.load(f)
    f.close()

    label = []
    file = []
    for key in data:
        file.append(key)
        label.append(convert_onehot(data[key]))
    label = np.vstack(label)

    return file, label

'''
This function matches labels with actual actions
Args:
    file_location: A string for the file location matching labels to actions
Returns:
    A dictionary with onehot labels (tuple) as keys and actions (string) as values
'''
def matching(file_location):
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.split(cur_dir)[0]
    f = open(parent_dir+'/dataset/'+file_location)
    data = json.load(f)
    f.close()

    match_dict = dict()
    for key in data:
        label = tuple(convert_onehot(int(key)))
        match_dict[label] = data[key]
    
    return match_dict

'''
This function extracts the first frame of video
Args:
    video_location: A string for the location of video to be captured
Returns:
    A 1d greyscale vector reshaped from 2d matrix (22528,)
Exception:
    First frame capture fails
'''
def extract_first_frame(video_location):
    vidcap = cv2.VideoCapture(video_location)
    success, image = vidcap.read()
    if success:
        grey_scale = np.mean(image,axis=-1)
        #img = np.repeat(np.expand_dims(grey_scale,-1),3,axis=-1)
        #cv2.imwrite("fuck.jpg",img)
        return grey_scale.reshape(-1)
    else:
        raise Exception("First frame not captured")

'''
This function extracts the first frame of all video data and saves as an npy file
Args:
    file: A list of file names returned from onehot_labels
Returns:
    A 2d numpy array of greyscale vectors (n,22528)
'''
def frame_to_data(file):
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.split(cur_dir)[0]
    dir = parent_dir + '/dataset/examples/'
    vid_end = '.mp4'

    img = []
    for name in file:
        img.append(extract_first_frame(dir+name+vid_end))

    np.save('img.npy', np.vstack(img))

    return np.vstack(img)

def main():

    file, label = onehot_labels('annotation_dict.json')
    #print(file)
    # data1 = matching("labels_dict.json")
    # print(data1)

    img = frame_to_data(file)
    print(img.shape)


if __name__ == '__main__':
    main()
