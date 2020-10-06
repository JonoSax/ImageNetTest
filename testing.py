import os
import itertools
import numpy as np
from glob import glob
import shutil
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator   


# source
src = "tiny-imagenet-200/"

# image datas
testDir = src + "test/"
trainDir = src + "train/"
valDir = src + "val/"
boxsDir = src + "boxes/"

# info
labelDir = src + "words.txt"

classes = os.listdir(valDir)      # get the class info from the directories


# get the id number and the associated label for all classes
labelsAll = np.array([[l[0], l[1].split("\n")[0]] for l in [l.split("\t") for l in open(labelDir).readlines()]])

# get the id number and label for only the classes in this dataset
labels = {}
while len(labels) < len(classes):
    for c in classes:
        pos = np.where(labelsAll[:, 0] == c)[0][0]

        labels[labelsAll[pos, 0]] = str(labelsAll[pos, 1])

# collect the data 


# create the data augmentation 
gen_Aug = ImageDataGenerator(                    
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True# ,
    # preprocessing_function=preprocess_input
)

# create the data generator WITH augmentation
train_generator = gen_Aug.flow_from_directory(
    trainDir,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='binary',
)

gen_noAug = ImageDataGenerator()
valid_generator = gen_Aug.flow_from_directory(
    valDir,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='binary',
)


model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input



# train the model 
r = model.fit(fx_train, fy_train, validation_data=(fx_test, fy_test), epochs=epoch)

