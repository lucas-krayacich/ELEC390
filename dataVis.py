import pandas as pd
import numpy as np
import matplotlib as mpl
import sklearn
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, \
    RocCurveDisplay, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import h5py
import math
import pickle
import joblib

mpl.use('TkAgg')  # !IMPORTANT

def noise_removal(df,window_size):
    denoised = df.copy()
    denoised = denoised.rolling(window=window_size).mean()
    denoised = denoised.dropna()
    return denoised


def extract_features(df,window_size):
    features = pd.DataFrame(columns=['maximum', 'minimum', 'range','mean', 'median', 'variance', 'skewness','std','kurtosis','label'])
    features['label']=df.iloc[:, -1]
    features['maximum'] = df.iloc[:, -2].rolling(window=window_size).max()
    features['minimum'] = df.iloc[:, -2].rolling(window=window_size).min()
    # features['range'] = df.iloc[:, -2].rolling(window=window_size).max()
    features['range'] = features['maximum'] - features['minimum']
    features['mean'] = df.iloc[:, -2].rolling(window=window_size).mean()
    features['median'] = df.iloc[:, -2].rolling(window=window_size).median()
    features['variance'] = df.iloc[:, -2].rolling(window=window_size).var()
    features['skewness'] = df.iloc[:, -2].rolling(window=window_size).skew()
    features['std'] = df.iloc[:, -2].rolling(window=window_size).std()
    features['kurtosis'] = df.iloc[:, -2].rolling(window=window_size).kurt()

    features = features.dropna()
    return features




#load the data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
LucasJumpRight = pd.read_csv('lucas files/Jump Right Labeled.csv')
LucasJumpLeft = pd.read_csv('lucas files/Jump Left Labeled.csv')
LucasWalkLeft = pd.read_csv('lucas files/Walk Left Labeled.csv')
LucasWalkRight = pd.read_csv('lucas files/Walk Right Labeled.csv')
########################################################
#plot for jump
# fig, ax = plt.subplots(figsize=(10,10))
# ax.plot(LucasJumpLeft.iloc[:,1].to_numpy(),linewidth=2)
# ax.plot(LucasJumpLeft.iloc[:,2].to_numpy(),linewidth=2)
# ax.plot(LucasJumpLeft.iloc[:,3].to_numpy(),linewidth=2)
# ax.plot(LucasJumpLeft.iloc[:,4].to_numpy(),linewidth=2)
# ax.set_title("acceleration vs time for jump")
# ax.legend(['Linear Acceleration x','Linear Acceleration y','Linear Acceleration z','Absolute acceleration'])
# ax.set_xlabel('time unit')
# ax.set_ylabel('acceleration m/s2')

#plot for walk
# fig, ax = plt.subplots(figsize=(10,10))
# ax.plot(LucasWalkLeft.iloc[:,1].to_numpy(),linewidth=2)
# ax.plot(LucasWalkLeft.iloc[:,2].to_numpy(),linewidth=2)
# ax.plot(LucasWalkLeft.iloc[:,3].to_numpy(),linewidth=2)
# ax.plot(LucasWalkLeft.iloc[:,4].to_numpy(),linewidth=2)
# ax.set_title("acceleration vs time for walk")
# ax.legend(['Linear Acceleration x','Linear Acceleration y','Linear Acceleration z','Absolute acceleration'])
# ax.set_xlabel('time unit')
# ax.set_ylabel('acceleration m/s2')

#basic datavis plots to d
#Walk y acceleration vs jump y acceleration -- maybe try this for a couple different acceleration dimensions
fig, ax = plt.subplots()

ax.plot(LucasJumpLeft.iloc[:,2].to_numpy())
ax.plot(LucasWalkLeft.iloc[:,2].to_numpy())
ax.set_title("Walk vs Jump Y acceleration")
ax.legend(['Jump Y accel', 'Walk Y accel'])
ax.set_xlabel('time unit')
ax.set_ylabel('acceleration m/s2')
plt.show()



fig, ax = plt.subplots()

ax.plot(LucasJumpLeft.iloc[:,1].to_numpy())
ax.plot(LucasWalkLeft.iloc[:,1].to_numpy())
ax.set_title("Walk vs Jump X acceleration")
ax.legend(['Jump X accel', 'Walk X accel'])
ax.set_xlabel('time unit')
ax.set_ylabel('acceleration m/s2')
plt.show()

fig, ax = plt.subplots()

ax.plot(LucasJumpLeft.iloc[:,4].to_numpy())
ax.plot(LucasWalkLeft.iloc[:,4].to_numpy())
ax.set_title("Walk vs Jump Abs acceleration")
ax.legend(['Jump Abs accel', 'Walk Abs accel'])
ax.set_xlabel('time unit')
ax.set_ylabel('acceleration m/s2')
plt.show()

import seaborn as sns



#########################################################################



LucasJumpLeftsma200 = noise_removal(LucasJumpLeft,50)
fig, ax = plt.subplots()
ax.plot(LucasJumpLeft.iloc[:,4].to_numpy())
ax.plot(LucasJumpLeftsma200.iloc[:,4].to_numpy())
ax.set_title("Jump vs Jump Denoised Abs acceleration -- Window size 50")
ax.legend(['Jump Abs accel', 'Jump Denoised Abs accel'])
ax.set_xlabel('time unit')
ax.set_ylabel('acceleration m/s2')
plt.show()
#drop time axis because its useless
LucasJumpLeftsma200 =LucasJumpLeftsma200.drop(LucasJumpLeftsma200.columns[0],axis=1)
LucasJumpLeftsma200Features = extract_features(LucasJumpLeftsma200,1000)
print(LucasJumpLeftsma200Features)

LucasJumpRightsma200 = noise_removal(LucasJumpLeft,200)
#drop time axis because its useless
LucasJumpLeftsma200 =LucasJumpLeftsma200.drop(LucasJumpLeftsma200.columns[0],axis=1)
LucasJumpLeftsma200Features = extract_features(LucasJumpLeftsma200,1000)
print(LucasJumpLeftsma200Features)



LucasWalkLeftSma200 = noise_removal(LucasWalkLeft,200)

LucasWalkLeftSma200 =LucasWalkLeftSma200.drop(LucasWalkLeftSma200.columns[0],axis=1)
LucasWalkLeftSma200Features = extract_features(LucasWalkLeftSma200,1000)
print(LucasWalkLeftSma200Features)

# LucasJumpLeftsma200 = noise_removal(LucasJumpLeft,50)
# fig, ax = plt.subplots()
# ax.plot(LucasJumpLeft.iloc[:,4].to_numpy())
# ax.plot(LucasJumpLeftsma200.iloc[:,4].to_numpy())
# ax.set_title("Jump vs Jump Denoised Abs acceleration -- Window size 50")
# ax.legend(['Jump Abs accel', 'Jump Denoised Abs accel'])
# ax.set_xlabel('time unit')
# ax.set_ylabel('acceleration m/s2')
# plt.show()
fig,ax = plt.subplots()
ax.plot(LucasJumpLeftsma200Features['skewness'])
ax.plot(LucasWalkLeftSma200Features['skewness'])
ax.set_title("Jump skewness vs Walk skewness")
ax.legend(['Jump skewness', 'Walk skewness'])
ax.set_xlabel('time unit')
ax.set_ylabel('skewness')
plt.show()

fig,ax = plt.subplots()
ax.plot(LucasJumpLeftsma200Features['kurtosis'])
ax.plot(LucasWalkLeftSma200Features['kurtosis'])
ax.set_title("Jump kurtosis vs Walk kurtosis")
ax.legend(['Jump kurtosis', 'Walk kurtosis'])
ax.set_xlabel('time unit')
ax.set_ylabel('kurtosis')
plt.show()

# x = LucasJumpLeftsma200Features['mean']
# x1 = LucasWalkLeftSma200Features['mean']
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
#
# ax1.scatter(x, y, s=10, c='b', marker="s", label='first')
# ax1.scatter(x,y, s=10, c='r', marker="o", label='second')
# plt.legend(loc='upper left')
# plt.show()
#
# test = sns.scatterplot(LucasWalkLeftSma200Features, LucasJumpLeftsma200Features, x='')


LucasWalkRightSma200 = noise_removal(LucasWalkRight,200)
LucasWalkRightSma200 =LucasWalkRightSma200.drop(LucasWalkRightSma200.columns[0],axis=1)
LucasWalkRightSma200Features = extract_features(LucasWalkRightSma200,1000)
print(LucasWalkRightSma200Features)

combined= pd.concat([LucasJumpLeftsma200Features,LucasWalkLeftSma200Features,LucasWalkRightSma200Features])
print(combined)
labels = combined.iloc[:, -1]
data = combined.iloc[:, 0:-1]

firstVis = sns.pairplot(combined)
maxRange = sns.jointplot(combined, x='maximum', y = 'range')

meanAndVariance = sns.jointplot(combined, x='mean', y='variance', color='black')
plt.show()
