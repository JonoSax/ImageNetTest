import os
from itertools import repeat
import numpy as np
from glob import glob
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.models import Model
import multiprocessing
from multiprocessing import Pool


def makeModel(modelType, IMAGE_SIZE, noClasses):

    # create the model topology
    # Inputs:   (modelType), text variable which leads to a different model which has been made below
    #           (IMAGE_SIZE), size of the images which are inputted
    #           (noClasses), number of classes (ie number of neurons to use)
    # Outputs:  (model), compiled model as constructd per modelType choice

    if modelType == "VGG19":

        model, preproFunc = VGG19Maker(IMAGE_SIZE, noClasses)

    elif modelType == "VGG16":

        model, preproFunc = VGG16Maker(IMAGE_SIZE, noClasses)

    elif modelType == "ResNet50":

        model, preproFunc = ResNet50Maker(IMAGE_SIZE, noClasses)

    elif modelType == "ResNet101":

        model, preproFunc = ResNet101Maker(IMAGE_SIZE, noClasses)

    elif modelType == "EfficientNetB7":

        model, preproFunc = EfficientNetB7Maker(IMAGE_SIZE, noClasses)

    # print the model toplogy 
    # model.summary()

    return(model, preproFunc)

## ------------------------ PREBUILD KERAS CNN MODELS ------------------------

def VGG16Maker(IMAGE_SIZE, noClasses, Trainable = False, Weights = 'imagenet', Top = False):

    # create a model with the VGG19 topology
    # Inputs:   (IMAGE_SIZE), size of the inputs
    #           (noClasses), number of classes the network will identify
    #           (Trainable), boolean whether the CNN component will be trainiable, defaults False
    #           (Weights), the pre-loaded weights that can be used, defaults to imagenet
    #           (Top), the dense layer which comes with the model, defaults to not being included
    # Outputs:  (model), compiled model

    # load the VGG16 and necessary layers from keras 
    from tensorflow.keras.applications import VGG16 as PretrainedModel
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.layers import Flatten, Dropout, Dense
    from tensorflow.keras.models import Model

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)         
            
    # boolean to fine-tune the CNN layers
    ptm.trainable = Trainable

    # create the dense layer
    x = denseLayer(ptm.output, noClasses)
            
    model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)

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

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)         
            
    # boolean to fine-tune the CNN layers
    ptm.trainable = Trainable

    # create the dense layer
    x = denseLayer(ptm.output, noClasses)
            
    model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)

def ResNet50Maker(IMAGE_SIZE, noClasses, Trainable = False, Weights = 'imagenet', Top = False):

    # create a model with the VGG19 topology
    # Inputs:   (IMAGE_SIZE), size of the inputs
    #           (noClasses), number of classes the network will identify
    #           (Trainable), boolean whether the CNN component will be trainiable, defaults False
    #           (Weights), the pre-loaded weights that can be used, defaults to imagenet
    #           (Top), the dense layer which comes with the model, defaults to not being included
    # Outputs:  (model), compiled model

    # load the ResNet and necessary layers from keras 
    from tensorflow.keras.applications import ResNet50 as PretrainedModel
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.layers import Flatten, Dropout, Dense
    from tensorflow.keras.models import Model

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)         
            
    # boolean to fine-tune the CNN layers
    ptm.trainable = Trainable

    # create the dense layer
    x = denseLayer(ptm.output, noClasses)
            
    model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)

def ResNet101Maker(IMAGE_SIZE, noClasses, Trainable = False, Weights = 'imagenet', Top = False):

    # create a model with the ResNet50 topology
    # Inputs:   (IMAGE_SIZE), size of the inputs
    #           (noClasses), number of classes the network will identify
    #           (Trainable), boolean whether the CNN component will be trainiable, defaults False
    #           (Weights), the pre-loaded weights that can be used, defaults to imagenet
    #           (Top), the dense layer which comes with the model, defaults to not being included
    # Outputs:  (model), compiled model

    # load the ResNet and necessary layers from keras 
    from tensorflow.keras.applications import ResNet101 as PretrainedModel
    from tensorflow.keras.applications.resnet import preprocess_input

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)         
            
    # boolean whether to fine-tune the CNN layers
    ptm.trainable = Trainable

    # create the dense layer
    x = denseLayer(ptm.output, noClasses)

    model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)

def EfficientNetB7Maker(IMAGE_SIZE, noClasses, Trainable = False, Weights = 'imagenet', Top = False):

    # create a model with the ResNet50 topology
    # Inputs:   (IMAGE_SIZE), size of the inputs
    #           (noClasses), number of classes the network will identify
    #           (Trainable), boolean whether the CNN component will be trainiable, defaults False
    #           (Weights), the pre-loaded weights that can be used, defaults to imagenet
    #           (Top), the dense layer which comes with the model, defaults to not being included
    # Outputs:  (model), compiled model

    # load the ResNet and necessary layers from keras 
    from tensorflow.keras.applications import EfficientNetB7 as PretrainedModel
    from tensorflow.keras.applications.efficientnet import preprocess_input

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)         
            
    # boolean whether to fine-tune the CNN layers
    ptm.trainable = Trainable

    # create the dense layer
    x = denseLayer(ptm.output, noClasses)

    model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)


## ---------------------------------------------------------------------------

def denseLayer(ptmoutput, noClasses):

    # creates the dense layer
    # Inputs:   (ptmoutput), takes the CNN output layer
    #           (noClasses), number of classifiers to train
    # Outputs:  (x), constructed dense layer

    # create the dense layer
    x = Flatten()(ptmoutput)   
    x = Dropout(0.2)(x)        
    x = Dense(noClasses, activation='softmax')(x)

    return(x)

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

def modelTrainer(src, modelName, gpu = 0, epochs = 100, batch_size = 64):
    
    # this function takes the compiled model and trains it on the imagenet data
    # Inputs:   (src), the data organised as test, train, val directories with each 
    #                   class organised as a directory
    #           (modelName), the specific model being used
    #           (gpu), the gpu being used. If no GPU is being used (ie on a CPU) then 
    #                   this variable is not important
    #           (epochs), number of training rounds for the model
    #           (batch_size), number of images processed for each round of parameter updates
    # Outputs:  (), atm no output but will eventually save the model to be used for 
    #                   evaluation

    # some basic info
    print("start, MODEL = " + modelName + " training on GPU " + str(gpu) + ", PID = " + str(os.getpid()))

    # set the GPU to be used
    CUDA_VISIBLE_DEVICES = gpu

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
    model, preproFunc = makeModel(modelName, IMAGE_SIZE, len(classes))

    # create the data augmentation 
    # NOTE I don't think this is augmenting the data beyond the number of samples
    # that are present.... 
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

    print("done, MODEL = " + modelName)

if __name__=="__main__":

    # load the cuda version needed for TF2 GPU usage
    os.system('module load cuda/cuda-10.1')
    print("\n------ Hardare acceleration: " + str(tf.test.is_gpu_available()) + " ------")
    print("------ TF version" + tf.__version__ + " ------\n")

    # stop the tf warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    # perform multiprocessing so that all the models can train at once
    multiprocessing.set_start_method("spawn")

    # set the source of the image data to train on
    src = "vtiny-imagenet-10/"

    # set the modesl which are available
    models = ["VGG19", "ResNet50", "ResNet101", "EfficientNetB7", "VGG16"]

    # specify the number of models you you want to train at a time
    # each model will occupy a single GPU if it is available
    gpuNo = 2

    # set the GPU number which each process will use (ensures that only n gpus are used 
    # rather than requiring another gpu for )
    GPUs = list(np.arange(gpuNo)) * int(np.ceil(len(models) / gpuNo))

    # train an individual model
    # modelTrainer(src, models[4], 0)

    # train multiple models at the same time
    with Pool(processes=gpuNo) as pool:
            pool.starmap(modelTrainer, zip(repeat(src), models, GPUs))


