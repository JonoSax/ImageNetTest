import os
import itertools
import numpy as np
from glob import glob
import shutil
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def makeModel(modelType, IMAGE_SIZE, noClasses):

    # create the model topology
    # Inputs:   (modelType), text variable which leads to a different model which has been made below
    #           (IMAGE_SIZE), size of the images which are inputted
    #           (noClasses), number of classes (ie number of neurons to use)
    # Outputs:  (model), full model as constructd per modelType choice

    from tensorflow.keras.optimizers import SGD, Adam
    from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    if modelType == "VGG19":

        from tensorflow.keras.applications import VGG19 as PretrainedModel

        # load the pretrained model and specify the weights being used
        ptm = PretrainedModel(
            input_shape=IMAGE_SIZE,   # create a 3D input (ie RGB)
            weights='imagenet',             # load in the specific imagenet weights
            include_top=False)              # load in only the CNN part
        
        # don't fine-tune the CNN layers
        ptm.trainable = False

        # create the dense layer
        x = Flatten()(ptm.output)   
        x = Dropout(0.2)(x)        
        x = Dense(noClasses, activation='softmax')(x)
        
        model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input

        # bolt the whole thing together, aka compile it
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    
    model.summary()

    return(model)


def classFinder(classPath, trainDir):

    # from the entire list of classes availabe in imagenet, find the 
    # ones used in this smaller set and their associated codes
    # Inputs:   (labelDir), path to the txt file with the info of all possible imagenet classes
    #           (valDir), directory of the training path
    # Outputs:  (labels), the code and its associated class as a dictionary

    classes = os.listdir(trainDir)      # get the class info from the directories

    # get the id number and the associated label for all classes
    labelsAll = np.array([[l[0], l[1].split("\n")[0]] for l in [l.split("\t") for l in open(classPath).readlines()]])

    # get the id number and label for only the classes in this dataset and create a dictionary
    labels = {}
    while len(labels) < len(classes):
        for c in classes:
            pos = np.where(labelsAll[:, 0] == c)[0][0]

            labels[labelsAll[pos, 0]] = str(labelsAll[pos, 1])


    return(labels)



def modelTrainer(src, epochs, batch_size):

    # image paths
    testDir = src + "test/"
    trainDir = src + "train/"
    valDir = src + "val/"

    classPath = src + "words.txt"

    # create a dictionary which will contain the classes and their codes being used
    classes = classFinder(classPath, trainDir)

    # get the paths of all the images + info
    image_files = glob(trainDir + '/*/*')       
    valid_image_files = glob(valDir + '/*/*')   
    Ntrain = len(image_files)
    Nvalid = len(valid_image_files)

    # get the image size (assumes all images are the same size)
    IMAGE_SIZE = list(cv2.imread(image_files[0]).shape)

    # create the model topology
    model = makeModel("VGG19", IMAGE_SIZE, len(classes))

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
        target_size=IMAGE_SIZE[:2],
        batch_size=batch_size,
        class_mode='binary',
    )

    # create the validating data generator (NO augmentation)
    gen_noAug = ImageDataGenerator()
    valid_generator = gen_Aug.flow_from_directory(
        valDir,
        target_size=IMAGE_SIZE[:2],
        batch_size=batch_size,
        class_mode='binary',
    )

    # train the model 
    r = model.fit(train_generator, validation_data=valid_generator, epochs=epochs)


if __name__=="__main__":


    src = "tiny-imagenet-200/"

    modelTrainer(src, 100, 64)