import pickle
import cv2
import numpy as np

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

model1=pickle.load(open('model.pkl','rb'))
model2=pickle.load(open('model1.pkl','rb'))
model3=pickle.load(open('model3.pkl','rb'))

from train import extract_color_histogram
from jpeg import decode
from train1 import minhash


def ensemble(imagep):
    image=cv2.imread(imagep)
    #cv2.imwrite('result.png',image)
    image1=cv2.resize(image,(128,128))
    hist = extract_color_histogram(image1)
    hist=np.array(hist)
    hist=np.expand_dims(hist,axis=0)
    pred1=int(model1.predict(hist)[0])

    hist1 = minhash(image1.flatten())
    hist1=np.array(hist1)
    hist1=np.expand_dims(hist1,axis=0)
    pred2=int(model2.predict(hist1)[0])

    f=decode(imagep)
    f=np.array(f)
    f=np.expand_dims(f,axis=0)
    pred3=int(model3.predict(f)[0])

    p=[pred1,pred2,pred3]
    p=np.array(p)
    return p




# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images('data'))
 
# initialize the data matrix and labels list
data = []
labels = []


if __name__=="__main__":


    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        break
        
        
        # load the image and extract the class label 
        print("imagesssssssssssssssssssssssssssssssssssssssss")
        print(imagePath)
        
        

        label=imagePath.split(os.path.sep)[-2]
        

        # extract a color histogram from the image, then update the
        # data matrix and labels list
        hist = ensemble(imagePath)
        data.append(hist)
        labels.append(label)


        
    # le = LabelEncoder()
    # labels = le.fit_transform(labels)

    


    # pickle.dump(data,open('data4.pkl','wb'))
    # pickle.dump(labels,open('labels4.pkl','wb'))

    data=pickle.load(open('data4.pkl','rb'))
    labels=pickle.load(open('labels4.pkl','rb'))



    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    print("[INFO] constructing training/testing split...")
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        np.array(data), labels, test_size=0.25, random_state=42)
    
    print(trainData.shape)
    print(trainLabels.shape) 
    # train the linear regression clasifier
   
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.ensemble import BaggingClassifier
    # from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier

    clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    model.fit(trainData, trainLabels)


    f = open('model5.pkl', "wb")
    f.write(pickle.dumps(model))
    f.close()
    predictions = model.predict(testData)

    print(classification_report(testLabels, predictions,
        ))






    
    





