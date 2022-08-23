import pickle
import cv2
import numpy as np

model1=pickle.load(open('model.pkl','rb'))
model2=pickle.load(open('model1.pkl','rb'))
model3=pickle.load(open('model3.pkl','rb'))
model4=pickle.load(open('model4.pkl','rb'))

from train import extract_color_histogram
from jpeg import decode
from train1 import minhash


def ensemble(imagep):
    image=cv2.imread(imagep)
    cv2.imwrite('result.png',image)
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
    p=np.expand_dims(p,axis=0)
    p1=model4.predict(p)
    return abs(p1-1)

    
    





