import cv2
from tqdm import tqdm
import glob
import numpy as np
import numpy.random as rand
import pickle
import tensorflow as tf


def get_data():
    folder="test"
    path1="binary_test/image"
    r=0
    bag=[]
    labels=[]
    la=0
    print("Resizing..")
    with tqdm(total=len(glob.glob(folder+"/*.png"))) as pbar:
        for img in glob.glob(folder+"/*.png"):
            '''
            if "cat" in img:
                l=np.array([1,0])
            elif "dog" in img:
                l=np.array([0,1])
            else:
                print("no label")
                continue
            '''
            im_gray = cv2.imread(img, 0)
            (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            im_bw=np.array(im_bw)
            im_bw=np.expand_dims(im_bw,axis=2)
            path=path1+str(r)+".png"     
            cv2.imwrite(path, im_bw)
            #labels.append(l)
            bag.append(im_bw)
            pbar.update(1)
            r+=1
    return bag,labels

X,y=get_data()

print("saving data..")
with open('binary_test_imgs', 'wb') as fp:
    pickle.dump(X, fp)
'''
with open('binary_train_labels', 'wb') as fp:
    pickle.dump(y, fp)
'''
print("Data saved")