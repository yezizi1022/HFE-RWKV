# -*- coding: utf-8 -*-

import h5py
import os
import numpy as np
import glob
from PIL import Image
# Parameters
height = 224
width = 224
channels = 3

############################################################# Prepare data set #################################################
Dataset_img = '/mnt/data/PSAX_US/Split/train'
Dataset_add = '/mnt/data/PSAX_US/Split_Label/train/'

img_list = glob.glob(os.path.join(Dataset_img,'*.png')) 
Tr_list = glob.glob(Dataset_add + '/*.png')
# It contains 2594 training samples
Data_train_2018 = np.zeros([7276, height, width, channels])
Label_train_2018 = np.zeros([7276, height, width])

print('Reading PSAX')
for idx in range(len(Tr_list)):
    if (idx + 1) % 100 == 0:
        print(f"{idx + 1}/{len(Tr_list)} have been precessed!!!!")

    img = Image.open(img_list[idx])
    img = img.resize((width,height),Image.NEAREST) 
    img = np.array(img ,np.float32) 
    Data_train_2018[idx, :, :, :] = img

#   img = sc.imread(Tr_list[idx])
#   img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode='RGB'))
  

#   b = Tr_list[idx]
#   a = b[0:len(Dataset_add)]
#   b = b[len(b) - 16: len(b) - 4]
#   add = (a + 'ISIC2018_Task1_Training_GroundTruth/' + b + '_segmentation.png')
    # img2 = sc.imread(Tr_list[idx])
    mask = Image.open(Tr_list[idx])
    mask = mask.resize((width,height),Image.NEAREST) 
    mask = np.array(mask ,np.float32) / 255
    # img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_2018[idx, :, :] = mask

print('Reading PSAX finished')

################################################################ Make the train and test sets ########################################
# We consider 1815 samples for training, 259 samples for validation and 520 samples for testing

Train_img = Data_train_2018
#Validation_img = Data_train_2018
#Test_img = Data_train_2018

Train_mask = Label_train_2018
#Validation_mask = Label_train_2018
#Test_mask = Label_train_2018

np.save('data_train', Train_img)
#np.save('data_test', Test_img)
#np.save('data_val', Validation_img)

np.save('mask_train', Train_mask)
#np.save('mask_test', Test_mask)
#np.save('mask_val', Validation_mask)
