import pickle
from sklearn
import pandas as pd
import numpy as np


#finding out all classes
df_train = pd.read_hdf("./features/mfcc_delta/timit_train.hdf")
print(df_train.head())
train_features = np.array(df_train["features"].tolist())
train_labels = np.array(df_train["labels"].tolist())
p = train_labels
p = list(set(train_labels))
p.sort()

#loading the models
models = []
for i in range(0,40):
    filename = "./mfcc_delta_delta/without_energy_coeff/model_of_"+p[i]+".pkl"
    f = open(filename, 'rb')
    m = pickle.load(f)         
    models.append(m)
    f.close() 
    print("Generated model for "+p[i])


df_test = pd.read_hdf("./features/mfcc_delta_delta/Test.hdf")
test_features = np.array(df_test["features"].tolist())#separating test feature
test_labels = np.array(df_test["labels"].tolist())#separating test labels

test_features=np.delete(test_features, [0,13,26], axis=1) #comment/uncomment according to with or without energy coeficient


#finding scores for every class
scores = np.empty([test_labels.shape[0],0])
for i in range(0,40):
    tmp = models[i].score_samples(test_features)
    scores = np.concatenate((scores,tmp.reshape(test_labels.shape[0],1)),axis=1)


#finding maximum score class == 40-way classification
pred_labels = np.argmax(scores,axis=1)
pred_labels2 = []
for i in range(0,pred_labels.shape[0]):
    pred_labels2.append(p[pred_labels[i]])


#calculating accuracy score
sklearn.metrics.accuracy_score(test_labels,pred_labels2)
    



