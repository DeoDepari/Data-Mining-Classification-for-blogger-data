# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:36:10 2020

@author: Deo Haganta Depari
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn import preprocessing

names = ['degree','caprice','topic','lmt','lpss','pb']

direktori='D:/Deo Haganta Depari/1.KAMPUS/Semester 4/Praktikum Data Mining dan Data Warehouse/Tugas Besar/data/kohkiloyeh.csv'
# membaca data dengan library panda
data = pd.read_csv(direktori,skiprows=1, names=names)

df = pd.DataFrame({
        'Degree': data['degree'],
        'Caprice': data['caprice'],
        'Topic': data['topic'],
        'LMT': data['lmt'],
        'LPSS': data['lpss']
        })


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
le=preprocessing.LabelEncoder()
dg_en = le.fit_transform(df.Degree)
#high=0;low=1;med=2

cp_en = le.fit_transform(df.Caprice)
#left=0;middle=1;rigth=2

tp_en = le.fit_transform(df.Topic)
#impression=0;news=1;political=2;scientific=3;tourism=4

lmt_en = le.fit_transform(df.LMT)
#no=0;yes=1

lpss_en = le.fit_transform(df.LPSS)
#no=0;yes=1

pb_en = le.fit_transform(names.LPSS)

print("K MEANS DENGAN N=3")
kmeans= KMeans(n_clusters=3)
kmeans.fit(df)

#label cluster yang sudah terbentuk
labels=kmeans.predict(df)
print("\nlabels:\n",labels)

#centroid yang sudah convergence.
centroids=kmeans.cluster_centers_
print("\nCentroids\n", centroids)

#banyak iterasi untuk mencapai convergence
n_iter=kmeans.n_iter_
print("\nBanyak Iterasi:", n_iter,"\n\n")

#Evaluasi Calinski Harabasz Index K=2-K=10
print("Evaluasi Calinski Harabasz Index N=2-N=10")
for i in range(2,11):
    clustering = AgglomerativeClustering(n_clusters=i, linkage='ward').fit(df)
    labels = clustering.labels_
    CHindex = metrics.calinski_harabasz_score(df, labels)
    print("CH Index untuk k = ",i," = ",CHindex)

print("\n\nK MEANS DENGAN N OPTIMUM=2")
kmeans= KMeans(n_clusters=2)
kmeans.fit(df)

#label cluster yang sudah terbentuk
labels=kmeans.predict(df)
print("\nlabels:\n",labels)

#centroid yang sudah convergence.
centroids=kmeans.cluster_centers_
print("\nCentroids\n", centroids)

#banyak iterasi untuk mencapai convergence
n_iter=kmeans.n_iter_
print("\nBanyak Iterasi:", n_iter,"\n\n")

#mengambil label dari k optimum
labeloptimum=labels

#Lakukan Evaluasi Clustering dengan FMI index
#mengevaluasi label hasil clustering
#ambil label asli dari data 
label_asli=data['class']
label_prediksi=labeloptimum
FMI=metrics.fowlkes_mallows_score(label_asli, label_prediksi)
print("FMI =", FMI)


print("\nEVALUASI DENGAN N=2-N=11")
#Lakukan Evaluasi Clustering dengan FMI index
#mengevaluasi label hasil clustering
#ambil label asli dari data
for i in range(2,11):
    clustering = AgglomerativeClustering(n_clusters=i, linkage='ward').fit(df)
    labels = clustering.labels_
    CHindex = metrics.calinski_harabasz_score(df, labels)
    print("CH Index untuk k = ",i," = ",CHindex)   
    
    label_asli=data['class']
    label_prediksi=labels
    FMI=metrics.fowlkes_mallows_score(label_asli, label_prediksi)
    print("FMI dengan n_cluster=",i,"=", FMI,"\n")
