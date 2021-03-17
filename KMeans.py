import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.preprocessing import Normalizer, MinMaxScaler
from math import sqrt


class KMeansModel():

    def __init__(self, K_clusters, X):
        self.K = K_clusters
        self.X = X
        self.Centres = np.random.rand(self.K, self.X.shape[1])
        self.Normalize()
        self.Classes = None
        self.N_Samples = self.X.shape[0]

    def GetEudlideanDistance(self, v1, v2):
        return sqrt(np.sum((v1-v2)**2))

    def FindClosestCentre(self, v1):
        ClosestCentreDistance = float("inf")
        ClosestCentreIndex = None
        for i in range(self.Centres.shape[0]):
            distance = self.GetEudlideanDistance(v1, self.Centres[i])
            if distance < ClosestCentreDistance:
                ClosestCentreDistance = distance
                ClosestCentreIndex = i
        return ClosestCentreIndex

    def MapClosestCentre(self):
        Vec = []
        for i in range(self.N_Samples):
            Vec.append(self.FindClosestCentre(self.X[i]))

        self.Classes = np.array(Vec).reshape((-1, 1))

    def Normalize(self):
        N = MinMaxScaler()
        self.X = N.fit_transform(self.X)

    def ChangeCentres(self):
        for i in range(self.Centres.shape[0]):
            self.Centres[i] = np.sum(
                self.X[np.array(km.Classes == i).reshape(1, -1)[0]], axis=0) / len(self.Classes[self.Classes == i])


TrainingX, TrainingY = make_blobs(
    n_samples=100, n_features=2, cluster_std=0.3, centers=2)
km = KMeansModel(2, TrainingX)
km.MapClosestCentre()


for i in range(50):
    km.ChangeCentres()
data = np.concatenate((km.X, km.Classes), axis=1)


plt.scatter(data[:, : -2], data[:, 1: -1], c="r")
plt.scatter(km.Centres[:, [0]], km.Centres[:, [1]], c="b")
plt.show()
