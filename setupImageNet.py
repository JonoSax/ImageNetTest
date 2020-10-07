import os
import numpy as np
from glob import glob
import multiprocessing
from multiprocessing import Pool


def movevalfiles(src):

    # create the validation directory structure as seen in the readme
    # reads the validation info and finds the associated image and moves it
    # into the correct class
    # Inputs:   (src), the path of the entire data collection
    # Outputs:  (), moves the files

    print("\n----- Moving validation files -----\n")

    valDir = src + "val/"

    # move the validation info out to the src
    os.system("mv " + valDir + "val_annotations.txt " + src)

    # the directory of the bouding boxes of the annotations
    labelDir = src + 'val_annotations.txt'

    # create a list which corresponds the image name with its category
    valInfo = np.array([[l[0], l[1].split("\n")[0]] for l in [l.split("\t") for l in open(labelDir).readlines()]])

    # move every image into the correct directory based on its class
    for n, (i, v) in enumerate(valInfo):
        v += "/"
        if n%100 == 0:
            print(str(n) + " images moved")
        try:
            os.mkdir(valDir + v)
        except:
            pass
        os.system("mv " + valDir + "images/" + i + " " + valDir + v)

    # remove the images folder which is now empty
    os.system("rm -r " + valDir + "images")


def moveimgs(c):

    # function which points to a single class of images and moves every 
    # image out of it
    # Inputs:   (c), the path of the class

    imgdir = c + "/images"

    imgs = glob(imgdir + "/*")
    for i in imgs:
        os.system("mv " + i + " " + c)

    try:
        os.rmdir(imgdir)
    except:
        pass

    c = c.split("/")[-1]
    print("Moved " + c)

def movetrainfiles(src):

    # create the training directory structure as seen in the readme
    # Inputs:   (src), the path of the entire data collection
    # Outputs:  (), moves the files

    print("\n----- Moving training files -----\n")

    traindir = src + "train/"
    boxdir = src + "boundingboxes/"

    # create a directory for the bounding box info to be moved to
    try:
        os.mkdir(boxdir)
    except:
        pass

    # move all the txt files to a new directory
    boxes = glob(traindir + "/*/*.txt")
    for b in boxes:
        os.system("mv " + b + " " + boxdir)

    # get the directories which point to the training images
    classes = sorted([traindir + f for f in os.listdir(traindir)])
    
    # serialised movement of images
    '''
    for c in classes:

        moveimgs(c)
    '''

    # move the images directy into the class directory
    # this is parallelised to use all but one of the available cores to speed this up
    with Pool(processes=(os.cpu_count()-1)) as pool:
        pool.starmap(moveimgs, zip(classes))


def movetestfiles(src):

    # move the test images out of the images directory

    print("\n----- Moving testing files -----\n")

    moveimgs(src + "test/")


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    src = "/Users/jonathanreshef/Downloads/tiny-imagenet-200-2/"

    movetrainfiles(src)

    movevalfiles(src)    

    movetestfiles(src)
