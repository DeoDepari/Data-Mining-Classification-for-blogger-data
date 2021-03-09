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
import sys
from sklearn.model_selection import cross_val_score

stdoutOrigin=sys.stdout 
sys.stdout = open("lOG_METODE_KNN_DISTANCE_TestSize=0.2.txt", "w")
for i in range(1,81):
    print("------------------------")
    print("N_Neighbors=",i)
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
    

    #divide data into data training&testing
    j=0.2
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=j, random_state=0)
    print("Test Size= ",j)
    neigh = KNeighborsClassifier(n_neighbors=i, weights="distance")
    neigh.fit(X_train, y_train)
    
    #print prediction
    print("Hasil klasifikasi menggunakan K-NN : ", neigh.predict(X_test))
    print()
    
    #print category(class) from real data
    print("Hasil klasifikasi data yang benar : ", y_test)
    print()
    
    # #print probability from data testing
    # print("Nilai Probabilitas tiap kategori (class) : ", neigh.predict_proba(X_test))
    # print()
    
    #getting prediction from data testing
    y_pred = neigh.predict(X_test)
    
    #print presentation of prediciton error
    error = ((y_test != y_pred).sum()/len(y_pred))*100
    print("Error prediksi = %.2f" %error,"%")
    
    #print presentation of accuracy
    akurasi = 100-error
    print("Akurasi = %.2f" %akurasi,"%")

    print("------------------------")
    
sys.stdout.close()
sys.stdout=stdoutOrigin