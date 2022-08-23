from scipy.spatial.distance import cosine
from random import randint
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
#from sklearn.externals import joblib
import pickle

# specify the length of each minhash vector
N = 128
max_val = (2**32)-1

# create N tuples that will serve as permutation functions
# these permutation values are used to hash all input sets
perms = [ (randint(0,max_val), randint(0,max_val)) for i in range(N)]

# initialize a sample minhash vector of length N
# each record will be represented by its own vec
vec = [float('inf') for i in range(N)]

def minhash(s, prime=4294967311):
  '''
  Given a set `s`, pass each member of the set through all permutation
  functions, and set the `ith` position of `vec` to the `ith` permutation
  function's output if that output is smaller than `vec[i]`.
  '''
  # initialize a minhash of length N with positive infinity values
  vec = [float('inf') for i in range(N)]

  for val in s:

    # ensure s is composed of integers
    if not isinstance(val, int): val = hash(val)

    # loop over each "permutation function"
    for perm_idx, perm_vals in enumerate(perms):
      a, b = perm_vals

      # pass `val` through the `ith` permutation function
      output = (a * val + b) % prime

      # conditionally update the `ith` value of vec
      if vec[perm_idx] > output:
        vec[perm_idx] = output

  # the returned vector represents the minimum hash of the set s
  return vec

if __name__=="__main__":

    # grab the list of images that we'll be describing
    print("[INFO] describing images...")
    imagePaths = list(paths.list_images('data'))
    
    # initialize the data matrix and labels list
    data = []
    labels = []


    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        break
        
        # load the image and extract the class label 
        print("imagesssssssssssssssssssssssssssssssssssssssss")
        print(imagePath)
        image = cv2.imread(imagePath)
        image=cv2.resize(image,(128,128))
        label=imagePath.split(os.path.sep)[-2]
        
        # data matrix and labels list
        hist = minhash(image.flatten())
        data.append(hist)
        labels.append(label)
        # break


        
    # le = LabelEncoder()
    # labels = le.fit_transform(labels)


    # pickle.dump(data,open('data1.pkl','wb'))
    # pickle.dump(labels,open('labels1.pkl','wb'))

    data=pickle.load(open('data1.pkl','rb'))
    labels=pickle.load(open('labels1.pkl','rb'))



    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    print("[INFO] constructing training/testing split...")
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        np.array(data), labels, test_size=0.25, random_state=42)
    

    print(trainData.shape)
    print(trainLabels.shape) 
    # train the linear regression clasifier
    print("[INFO] training Linear SVM classifier...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(trainData, trainLabels)


    f = open('model1.pkl', "wb")
    f.write(pickle.dumps(model))
    f.close()


    
