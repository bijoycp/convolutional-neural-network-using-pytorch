import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
# mixing up or currently ordered data that might lead our network astray in training.
from random import shuffle

path = 'data'

IMG_SIZE = 96

nb_classes = 15


def create_train_data():
    label_dict={}
    training_data = []
    label = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        for dirname in dirnames:
            print(dirname)
            for(direcpath, direcnames, files) in os.walk(path + "/" + dirname):
                for file in files:
                    actual_path = path + "/" + dirname + "/" + file
                    print(files)
                    # label=label_img(dirname)
                    path1 = path + "/" + dirname + '/' + file
                    img = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    training_data.append(
                        [np.array(img), np.eye(nb_classes)[label]])
                    # [np.array(img), np.eye(2)[self.LABELS[label]]])
            label_dict.update( {label : dirname} )
            label = label + 1
            print(label)
    shuffle(training_data)
    np.save('train_data_1.npy', training_data)
    print(training_data)
    return training_data,label_dict


training_data,label_dict = create_train_data()
print(label_dict)

