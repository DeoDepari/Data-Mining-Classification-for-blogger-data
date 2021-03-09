# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:50:49 2020

@author: irzar
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import itertools

names = ['degree','caprice','topic','lmt','lpss','pb']
dataset = pd.read_csv("kohkiloyeh.csv", names=names)
arrayData = dataset.values

#convert category from string to numeric
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
pb_en = le.fit_transform(pbs)
#no=0;yes=1

#Seperate features and class (category)
fitur_gabung = np.array([dg_en,cp_en,tp_en,lmt_en,lpss_en])
X_data = np.ndarray.transpose(fitur_gabung)
y_data = pb_en

#divide data into data training&testing
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)

model = KNeighborsClassifier(n_neighbors=7, weights="distance")
model.fit(X_train, y_train)

#getting prediction from data testing
predicted = model.predict(X_test)

#print prediction
print("Hasil klasifikasi menggunakan K-NN : \n", predicted)
print()

#print category(class) from real data
print("Hasil klasifikasi data yang benar : \n", y_test)
print()

# print probability from data testing
# print("Nilai Probabilitas tiap kategori (class) : \n", model.predict_proba(X_test))
# print()

#print presentation of prediciton error
error = ((y_test != predicted).sum()/len(predicted))*100
print("Error prediksi   = %.2f" %error,"%")

#print presentation of accuracy
akurasi = 100-error
print("Akurasi          = %.2f" %akurasi,"%")

#evaluasi model
def Conf_matrix(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_actual[i] !=y_pred[i]:
            FP += 1
        if y_actual[i]==y_pred[i]==0:
            TN += 1
        if y_pred[i]==0 and y_actual[i]!= y_pred[i]:
            FN += 1           
    return (TP, FN, TN, FP)

#hold out

TP, FN, TN, FP = Conf_matrix(y_test, predicted)

print('\n\nEvaluasi Model Hold Out Estimation :')
print('akurasi      = ', (TP+TN)/(TP+TN+FP+FN))
print('sensitivity  = ', TP/(TP+FN))
print('specificity  = ', TN/(TN+FP))

#kfold
prediksi = list()
harapan = list()

skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X_data, y_data):        
        prediksi.append(predicted)
        harapan.append(y_test)
        
pred = list(itertools.chain.from_iterable(prediksi))
harap = list(itertools.chain.from_iterable(harapan))

TP1, FN1, TN1, FP1 = Conf_matrix(harap, pred)

print('\n\nEvaluasi Model K-Fold Cross Validation :')
print('akurasi      = ', (TP1+TN1)/(TP1+TN1+FP1+FN1))
print('sensitivity  = ', TP1/(TP1+FN1))
print('specificity  = ', TN/(TN1+FP1))