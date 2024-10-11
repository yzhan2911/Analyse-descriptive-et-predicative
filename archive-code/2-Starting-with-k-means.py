"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="flame.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
"""
k_values = []
silhouette_scores = []
inertias = []
times = []
for k in range(2,50):
    tps1 = time.time()
    #k=7
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=15)
    model.fit(datanp)
    tps2 = time.time()
    #model2= cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    #model2.fit(datanp)
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_
    silhouette_avg = metrics.silhouette_score(datanp, labels)
    k_values.append(k)
    silhouette_scores.append(silhouette_avg)
    inertias.append(inertie)
    times.append(tps2 - tps1)
    print(f"For k={k}, sildonut2houette score={silhouette_avg}, inertia={inertie}, time taken={tps2 - tps1}")

fig, ax1 = plt.subplots(3, 1, figsize=(10, 18))

ax1[0].plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')
ax1[0].set_ylabel('Coefficient de Silhouette')
ax1[0].grid(True)


ax1[1].plot(k_values, inertias, marker='o', linestyle='-', color='g')
ax1[1].set_ylabel('Inertie')
ax1[1].grid(True)

ax1[2].plot(k_values, times, marker='o', linestyle='-', color='r')
ax1[2].set_xlabel('Number of clusters (k)')
ax1[2].set_ylabel('Time(ms)')
ax1[2].grid(True)
plt.tight_layout()
plt.show()
"""
k=2
tps1 = time.time()
model2= cluster.KMeans(n_clusters=k, init='random', n_init=2)
model2.fit(datanp)
tps2 = time.time()
labels = model2.labels_
iteration = model2.n_iter_
inertie = model2.inertia_
centroids2 = model2.cluster_centers_ 
plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids2[:, 0],centroids2[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
print("labels", labels)

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids2)
print(dists)

