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
    A dictionary with file names as keys and onehot labels (np.array) as values
'''
def onehot_labels(file_location):
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.split(cur_dir)[0]
    f = open(parent_dir+'/dataset/'+file_location)
    data = json.load(f)
    f.close()
    
    for key in data:
        label = data[key]
        data[key] = convert_onehot(label)

    return data

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
This function extracts the first frame of video in 2D matrix
Args:
    video_location: A string for the location of video to be captured
Returns:
    A 2d matrix of extracted first frame image
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
    None
Returns:
    A dictionary with file names (string) as keys and 2d image matrix as values
'''
def frame_to_data():
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.split(cur_dir)[0]
    dir = parent_dir+'/dataset/examples/'
    vid_end = '.mp4'
    img_dict = dict()
    i = 0
    for vid_file in os.scandir(dir):
        if i > 2000:
            break
        if vid_file.name.endswith(vid_end):
            img_name = vid_file.name[:len(vid_file.name)-4]
            img_dict[img_name] = extract_first_frame(dir+vid_file.name)
            i += 1
    
    np.save('img_dict.npy',img_dict)
    
    return img_dict


def main():
    '''
    data = onehot_labels('annotation_dict.json')
    print(data)
    data1 = matching("labels_dict.json")
    print(data1)
    '''
    img = frame_to_data()
    print(img)
    

if __name__ == '__main__':
    main()
