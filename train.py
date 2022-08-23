

#python train.py --dataset dataset --labelbin model.pickle

# import the necessary packages
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


 
# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images('data'))
 
# initialize the data matrix and labels list
data = []
labels = []


def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
 
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
 
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
 
	# return the flattened histogram as the feature vector
	return hist.flatten()


if __name__=="__main__":


	# loop over the input images
	for (i, imagePath) in enumerate(imagePaths):
		
		# load the image and extract the class label 
		print("imagesssssssssssssssssssssssssssssssssssssssss")
		print(imagePath)
		image = cv2.imread(imagePath)
		image=cv2.resize(image,(128,128))
		

		label=imagePath.split(os.path.sep)[-2]
		
	
		# extract a color histogram from the image, then update the
		# data matrix and labels list
		hist = extract_color_histogram(image)
		data.append(hist)
		labels.append(label)


		
	le = LabelEncoder()
	labels = le.fit_transform(labels)


	pickle.dump(data,open('data.pkl','wb'))
	pickle.dump(labels,open('labels.pkl','wb'))

	data=pickle.load(open('data.pkl','rb'))
	labels=pickle.load(open('labels.pkl','rb'))



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


	f = open('model.pkl', "wb")
	f.write(pickle.dumps(model))
	f.close()


	



