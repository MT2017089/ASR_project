import pandas as pd
import numpy as np
import sklearn
from sklearn import mixture
import pickle

df = pd.read_hdf("./features/mfcc/Train.hdf")#change path to appropriate train
features_train = np.array(timit_df["features"].tolist())
labels_train = np.array(timit_df["labels"].tolist())
p = list(set(labels))
p.sort()

feature_list = []
for k in p:
    feature_list.append(features_train[labels_train == k])


def model(no_of_mix,path):#creating model function 
    for i in range(0,40):
        gaus_mix = mixture.GaussianMixture(n_comp = no_of_mix,200,covariance_type='diag',n_init = 3)
        gaus_mix.fit(feature_list[i])
        filename = path+"model_of_"+p[i]+".pkl"
        pickle.dump(gaus_mix, open(filename, 'wb'))

model(64,"./models/mfcc/with_energy_coeff/")

