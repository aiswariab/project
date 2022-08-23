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
from jpeg import decode

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
    feat=decode(imagePath)
    


    label=imagePath.split(os.path.sep)[-2]


    # extract a color histogram from the image, then update the
    # data matrix and labels list
   
    data.append(feat)
    labels.append(label)

    


	
# le = LabelEncoder()
# labels = le.fit_transform(labels)


# pickle.dump(data,open('data3.pkl','wb'))
# pickle.dump(labels,open('labels3.pkl','wb'))

data=pickle.load(open('data3.pkl','rb'))
labels=pickle.load(open('labels3.pkl','rb'))





# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.25, random_state=42)
 

print(trainData.shape)
print(trainLabels.shape) 
# train the linear regression clasifier
print("[INFO] training Linear SVM classifier...")
import lightgbm as lgb
from lightgbm import LGBMClassifier
model = LGBMClassifier()
model.fit(trainData, trainLabels)


f = open('model3.pkl', "wb")
f.write(pickle.dumps(model))
f.close()



