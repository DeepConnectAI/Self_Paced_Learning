# run using cmd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X = pd.read_csv("./../data_sets/hardwork_pays_off/TrainingData/Linear_X_Train.csv").values
Y = pd.read_csv("./../data_sets/hardwork_pays_off/TrainingData/Linear_Y_Train.csv").values

theta = np.load("./theta_list.npy")

t0 = theta[:,0]
t1 = theta[:,1]

plt.ion()
for i in range(100):
    y_pred = t1[i]*X + t0[i]
    # orginal line
    plt.scatter(X,Y)
    # predicted line
    plt.plot(X,y_pred,'red')
    plt.draw()
    plt.pause(0.5)
    plt.clf()
    
    