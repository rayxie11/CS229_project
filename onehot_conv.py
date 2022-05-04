# This function converts the labelling in dataset into onehot notations
import os
import json
import numpy as np

def load_data(file_location):
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
    name, label = load_data('annotation_dict.json')
    print(name)
    print(label)

if __name__ == '__main__':
    main()
