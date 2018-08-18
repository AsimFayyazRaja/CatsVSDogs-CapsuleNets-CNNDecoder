import cv2
from tqdm import tqdm
import glob
import numpy as np
import numpy.random as rand
import pickle
import tensorflow as tf


def get_data():
	folder="Data/test"
	path1="64_test/"
	r=0
	bag=[]
	labels=[]
	la=0
	print("Resizing..")
	with tqdm(total=len(glob.glob(folder+"/*.jpg"))) as pbar:
		for img in glob.glob(folder+"/*.jpg"):
			path1="64_test/"
			if "cat" in img:
				l=np.array([1,0])
				path1=path1+"cat"
			elif "dog" in img:
				l=np.array([0,1])
				path1=path1+"dog"
			n = np.array(cv2.imread(img))
			resized_image = np.array(cv2.resize(n, (64, 64)))
			resized_image = np.array(resized_image, dtype=np.float32) / 255 - 0.5
			path=path1+str(r)+".png"
			#cv2.imwrite(path, resized_image)
			#labels.append(l)
			bag.append(resized_image)
			pbar.update(1)
			r+=1
	return bag,labels

X,y=get_data()

print("saving data..")
with open('64_test_imgs', 'wb') as fp:
    pickle.dump(X, fp)

with open('64_test_labels', 'wb') as fp:
    pickle.dump(y, fp)

print("Data saved")
