import os
import numpy as np


def movevalfiles():
    # this script moves the validation images into new directories based 
    # on their label
    
    src = "tiny-imagenet-200/"

    valdir = src + "val/"

    labelDir = src + 'val_annotations.txt'

    valInfo = np.array([[l[0], l[1].split("\n")[0]] for l in [l.split("\t") for l in open(labelDir).readlines()]])

    for i, v in valInfo:
        v += "/"
        print(i + " " + v)
        try:
            os.mkdir(valdir + v)
        except:
            pass
        os.system("mv " + valdir + i + " " + valdir + v)

if __name__ == "__main__":

    movevalfiles()    