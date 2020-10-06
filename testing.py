import os
import itertools
import numpy as np
from glob import glob
import shutil
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("\nHardare acceleration: " + str(tf.test.is_gpu_available()) + "\n")
print("\nTF version" + tf.__version__ + "\n")


def makeModel(modelType, IMAGE_SIZE, noClasses):

    # create the model topology
    # Inputs:   (modelType), text variable which leads to a different model which has been made below
    #           (IMAGE_SIZE), size of the images which are inputted
    #           (noClasses), number of classes (ie number of neurons to use)
    # Outputs:  (model), compiled model as constructd per modelType choice

    if modelType == "VGG19":

        model, preproFunc = VGG19Maker(IMAGE_SIZE, noClasses)
    
    # print the model toplogy 
    model.summary()

    return(model, preproFunc)

def VGG19Maker(IMAGE_SIZE, noClasses, Trainable = False, Weights = 'imagenet', Top = False):

    # create a model with the VGG19 topology
    # Inputs:   (IMAGE_SIZE), size of the inputs
    #           (noClasses), number of classes the network will identify
    #           (Trainable), boolean whether the CNN component will be trainiable, defaults False
    #           (Weights), the pre-loaded weights that can be used, defaults to imagenet
    #           (Top), the dense layer which comes with the model, defaults to not being included
    # Outputs:  (model), compiled model

    # load the VGG19 and necessary layers from keras 
    from tensorflow.keras.applications import VGG19 as PretrainedModel
    from tensorflow.keras.applications.vgg19 import preprocess_input
    from tensorflow.keras.layers import Flatten, Dropout, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)         
            
    # boolean to fine-tune the CNN layers
    ptm.trainable = Trainable

    # create the dense layer
    x = Flatten()(ptm.output)   
    x = Dropout(0.2)(x)        
    x = Dense(noClasses, activation='softmax')(x)
            
    model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam,
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)

def classFinder(classPath, trainDir, noClasses = -1):

    # from the entire list of classes availabe in imagenet, find the 
    # ones used in this smaller set and their associated codes
    # Inputs:   (labelDir), path to the txt file with the info of all possible imagenet classes
    #           (valDir), directory of the training path
    #           (noClasses), number of classes to train, defaults to 0 (aka use all availabe classes)
    # Outputs:  (labels), the code and its associated class as a dictionary

    # get the class info from the directories
    classes = sorted(os.listdir(trainDir))
    if noClasses != -1:
        classes = classes[:noClasses]   

    # get the id number and the associated label for all classes
    labelsAll = np.array([[l[0], l[1].split("\n")[0]] for l in [l.split("\t") for l in open(classPath).readlines()]])

    # get the id number and label for only the classes in this dataset and create a dictionary
    labels = {}

    # find the corresponding classes in the full imagenet catalogue, add its coresponding
    # ID to the dictionary
    for c in classes:
        try:
            pos = np.where(labelsAll[:, 0] == c)[0][0]
            labels[labelsAll[pos, 0]] = str(labelsAll[pos, 1])
        except:
            pass



    return(labels)

def modelTrainer(src, epochs, batch_size):

    # info paths
    testDir = src + "test/"
    trainDir = src + "train/"
    valDir = src + "val/"

    # the path of the full imagenet catalogue info
    classPath = src + "words.txt"

    # create a dictionary which will contain the classes and their codes being used
    classes = classFinder(classPath, trainDir)


    # get the image size (assumes all images are the same size)
    validImages = glob(valDir + "*/*")
    IMAGE_SIZE = list(cv2.imread(validImages[0]).shape)

    # create the model topology
    model, preproFunc = makeModel("VGG19", IMAGE_SIZE, len(classes))

    # create the data augmentation 
    gen_Aug = ImageDataGenerator(                    
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preproFunc
    )

    # create the data generator WITH augmentation
    train_generator = gen_Aug.flow_from_directory(
        trainDir,
        target_size=IMAGE_SIZE[:2],     # only use the first two dimensions of the images
        batch_size=batch_size,
        class_mode='binary',
    )

    # create the validating data generator (NO augmentation)
    gen_noAug = ImageDataGenerator(preprocessing_function=preproFunc)
    valid_generator = gen_Aug.flow_from_directory(
        valDir,
        target_size=IMAGE_SIZE[:2],
        batch_size=batch_size,
        class_mode='binary',
    )

    # train the model 
    r = model.fit(train_generator, validation_data=valid_generator, epochs=epochs)


if __name__=="__main__":


    src = "vtiny-imagenet-10/"

    modelTrainer(src, 100, 64)