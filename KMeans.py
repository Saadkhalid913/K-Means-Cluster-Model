import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.preprocessing import Normalizer, MinMaxScaler
from math import sqrt
from sklearn.metrics import confusion_matrix, accuracy_score


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
        # loops through all centres and returns the index of closest centre
        ClosestCentreDistance = float("inf")
        ClosestCentreIndex = None
        for i in range(self.Centres.shape[0]):
            distance = self.GetEudlideanDistance(v1, self.Centres[i])
            if distance < ClosestCentreDistance:
                ClosestCentreDistance = distance
                ClosestCentreIndex = i
        return ClosestCentreIndex

    def MapClosestCentre(self):
        # This is done for each sample and creates a column vector where the 
        # element at index i of the vector corresponds to the closest centre of
        # the ith sample
        Vec = []
        for i in range(self.N_Samples):
            Vec.append(self.FindClosestCentre(self.X[i]))

        self.Classes = np.array(Vec).reshape((-1, 1))

    def Normalize(self):
        # Normalizing function to express values between 0 and 1 
        N = MinMaxScaler()
        self.X = N.fit_transform(self.X)

    def ChangeCentres(self):
        # updates each centre to be the take on the mean position of each 
        # sample which takes on the corresponding class of the centre
        for i in range(self.Centres.shape[0]): # we loop through all the centres 
            self.Centres[i] = np.sum(
                # We use boolean indexing to only take the rows which correspond to the ith centre (the one we are currently re-calculating) 
                self.X[np.array(km.Classes == i).reshape(1, -1)[0]], axis=0) / len(self.Classes[self.Classes == i]) # we divide by the number of samples in the class 

if __name__ == "__main__":
    # Creating some testing data 
    k=2
    TrainingX, TrainingY = make_blobs(
        n_samples=1000, n_features=10, cluster_std=4, centers=k)
    km = KMeansModel(k, TrainingX)
    
    km.MapClosestCentre()


    for i in range(50):
        km.ChangeCentres()
    data = np.concatenate((km.X, km.Classes), axis=1)
    km.MapClosestCentre()
    
    # only uncomment if n_features =2 
    # plt.scatter(data[:, : -2], data[:, 1: -1], c="r")
    # plt.scatter(km.Centres[:, [0]], km.Centres[:, [1]], c="b")
    # plt.show()
    # print(np.concatenate((km.X, km.Classes), axis=1))

    # this code either yields a very low or high accuracy in most cases, this is just
    # becuase this is an unsupervised learning model and that the names of classes 
    # are completely arbitrary. A better measure of accuracy the the confusion matrix, 
    # which shows classes and their frequencies 
    pred = km.Classes
    act = TrainingY.reshape(-1,1)
    print(np.concatenate((pred,act ), axis=1 ))
    print(confusion_matrix(pred,act ))
    print(accuracy_score(pred, act) *100,"%")
