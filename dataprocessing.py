import cv2
import os
import sys
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from os.path import join

def readf(input_dir, img_size, output_dir):
    try:
        tclass = [ d for d in os.listdir( input_dir ) ]
        counter = 0
        for x in tclass:
           try:
               img = cv2.imread(os.path.join(input_dir+'/',x))
               img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_AREA)
               cv2.imwrite(os.path.join(output_dir+'/', 'resized'+x),img)
               print "Resizing file : %s - %s " % (x,d)
           except Exception,e:
                print "Error resize file : %s - %s " % (x,d)
                sys.exit(1)
           counter +=1
    except Exception,e:
        print "Error, check Input directory etc : ", e
        sys.exit(1)

img_size = 64
for i in ['sandwich', 'sushi']:
    input_dir = "/media/shubhabrata/DATAPART1/sushi_or_sandwich_photos/sushi_or_sandwich/" + i
    output_dir = "/media/shubhabrata/DATAPART1/sushi_or_sandwich_photos/resized64/" + i
    readf(input_dir, img_size, output_dir)

"""
def reshapeImg(img):
    s = np.asarray(img)
    return s  #.transpose(2,0,1).reshape(3,-1)
"""   

## prepare data for trainig and testing

location_sandwich = "/home/shubhabrata/Desktop/CookPad/sandwich"
location_sushi = "/home/shubhabrata/Desktop/CookPad/sushi"

sdwdir = [ d for d in os.listdir(location_sandwich) ]

list_sdw = [cv2.imread(os.path.join(location_sandwich+'/',x)) for x in sdwdir]
label_sdw = [0]*len(list_sdw)

susdir = [ d for d in os.listdir(location_sushi) ]

list_sus = [cv2.imread(os.path.join(location_sushi+'/',x)) for x in susdir]
label_sus = [1]*len(list_sus)

df = pd.DataFrame({"data": list_sdw + list_sus , "label": label_sdw + label_sus})
df = shuffle(df)
df = df.reset_index(drop = True)

X = np.array(df['data'].tolist())
y = np.array(df['label'].tolist())

data_dir = "/home/shubhabrata/Desktop/CookPad/"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)   

np.save(join(data_dir, 'X_train.npy'), X_train) 
np.save(join(data_dir, 'X_test.npy'), X_test) 
np.save(join(data_dir, 'y_test.npy'), y_test) 
np.save(join(data_dir, 'y_train.npy'), y_train) 