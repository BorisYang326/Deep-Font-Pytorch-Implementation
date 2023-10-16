from __future__ import division
# import numpy as np
# import math
# import bcfstore
import util2
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm
# affine2d = __import__('affine2d')

SYN_TRAIN_FLAG = 0
SYN_VAL_FLAG = 1
REAL_TEST_FLAG = 2
# TO RUN THIS FILE YOU MUST HAVE HAD DOWNLOADED SYN_TRAIN AND LABELS.
# Files not in github as syn_train is upwards of 20GB.

def initialize(data_dir, test):
    """ data_dir: directory of samples
        test: directory of labels

        used to initialize the datastore and labelstore constructs of BCF
    """
    dataStore, labelStore = util2.read_bcf_file(data_dir,test)
    return dataStore, labelStore

AdobeVFR_root = '/public/dataset/AdobeVFR'
syn_train_path = '/BCF format/VFR_syn_train'
syn_val_path = '/BCF format/VFR_syn_val'
syn_train_savepath = AdobeVFR_root + '/Raw Image/VFR_syn_train/'
syn_val_savepath = AdobeVFR_root + '/Raw Image/VFR_syn_val/'
font_path = AdobeVFR_root + '/fontlist.txt'



def unbcf(bcf_path, savepath,font_book_path,dataset_type):
    """ This function is used to convert the BCF format to a folder of images
        in the same format as the IAM dataset.
    """
    # Iterate through each image in the datastore
    fonttxt = open(font_book_path, "r")
    fonttxt = fonttxt.read().split()
    fontlist = []
    for font in fonttxt:
        fontlist.append(font)
    count = 0
    tp_dataStore, tp_labelStore = initialize(bcf_path,dataset_type)
    for i in tqdm(range(0, tp_dataStore.size())):
        im = Image.open(BytesIO(tp_dataStore.get(i)))

        # get fontlist and for each index in label, add count to end of name
        index = tp_labelStore[i]
        if i > 0 and index == tp_labelStore[i-1]:
            count += 1
        else:
            count = 0

        fontname = fontlist[index]
        fontfile = fontname +'_'+ str(count)

        savepath_ = savepath + fontname
        # Make directory if it does not exist.
        if not os.path.exists(savepath_):
            os.makedirs(savepath_)

        im.save(savepath_ + "/" + fontfile + ".png", "PNG")

unbcf(AdobeVFR_root+syn_train_path,syn_train_savepath,font_path,SYN_TRAIN_FLAG)
unbcf(AdobeVFR_root+syn_val_path,syn_val_savepath,font_path,SYN_VAL_FLAG)