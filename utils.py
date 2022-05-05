'''
This file contains all the util functions that are needed to extract information from dataset
'''
import os
import json
import numpy as np

# convert labels [0,10] into onehot labels
def onehot_labels(file_location):
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.split(cur_dir)[0]
    f = open(parent_dir+'/dataset/'+file_location)
    data = json.load(f)
    
    data_name = []
    data_label = np.zeros((len(data),11))
    i = 0
    for key in data:
        data_name.append(key)
        data_label[i,data[key]] = 1
        i += 1

    return data_name, data_label

def main():
    name, label = onehot_labels('annotation_dict.json')

if __name__ == '__main__':
    main()
