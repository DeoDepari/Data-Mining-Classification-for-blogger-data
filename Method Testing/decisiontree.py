# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:50:56 2020

@author: Deo Haganta Depari
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sys

stdoutOrigin=sys.stdout 
sys.stdout = open("lOG_METODE_DECISIONTREE.txt", "w")

direktori='D:/Deo Haganta Depari/1.KAMPUS/Semester 4/Praktikum Data Mining dan Data Warehouse/Tugas Besar/dataset/kohkiloyeh.csv'
    
names = ['degree','caprice','topic','lmt','lpss','pb']
dataset = pd.read_csv(direktori,skiprows=1, names=names)
arrayData = dataset.values

degrees = dataset['degree']
caprices= dataset['caprice']
topics = dataset['topic']
lmts = dataset['lmt']
lpsss = dataset['lpss']
pbs = dataset['pb']

le = preprocessing.LabelEncoder()
dg_en = le.fit_transform(degrees)
#high=0;low=1;med=2
cp_en = le.fit_transform(caprices)
#left=0;middle=1;rigth=2
tp_en = le.fit_transform(topics)
#impression=0;news=1;political=2;scientific=3;tourism=4
lmt_en = le.fit_transform(lmts)
#no=0;yes=1
lpss_en = le.fit_transform(lpsss)
#no=0;yes=1

#Seperate features and class (category)
fitur_gabung = np.array([dg_en,cp_en,tp_en,lmt_en,lpss_en])
X_data = np.ndarray.transpose(fitur_gabung)
y_data = arrayData[:,5]

#convert category from string to numeric
y_data = le.fit_transform(y_data)

data_test=X_data[0:20]
data_train=X_data[20:100]

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()

clf=clf.fit(data_train,y_data[20:100])

jumlahbenar=0
for i in range(0,20):
    y_pred=clf.predict([data_test[i]])
    print("------------------------")
    print("Prediksi untuk X= ",y_pred)
    print("Kelas Sebenarnya=  ",y_data[i])
    if(y_pred==y_data[i]):
        print("Prediksi Benar")
        jumlahbenar+=1
    else:
        print("Prediksi Salah")
    print("------------------------")
print("Banykanya Prediksi yang benar",jumlahbenar)
sys.stdout.close()
sys.stdout=stdoutOrigin