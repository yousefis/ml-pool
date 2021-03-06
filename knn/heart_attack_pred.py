#%%


import csv
import json
import argparse
from os.path import join

import numpy as np
import requests
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

'''
Dataset description:
    age: age in years
    sex:0,1 male female
    cp: chest pain type
    trestbps: resting blood pressure (in mm Hg on admission to the hospital)
    chol: serum cholestoral in mg/dl
    fbs: (fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)
    restecg: resting electrocardiographic results
    thalach: maximum heart rate achieved
    exang: exercise induced angina (1 = yes; 0 = no)
    oldpeak: ST depression induced by exercise relative to rest
    slope: the slope of the peak exercise ST segment
    ca: number of major vessels (0-3) colored by flourosopy
    thal: maximum heart rate achieved, 3 = normal; 6 = fixed defect; 7 = reversable defect
    target: 0= less chance of heart attack 1= more chance of heart attack
'''

def distance(point1, point2):
    subt = [(p1-p2)**2 for (p1,p2) in zip(point1, point2)]
    return sum(subt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-path",type=str, help="path to the data",
                        default= "../datasets/")
    arg = parser.parse_args()
    csv_name = 'heart-attack-prediction.csv'

    df = pd.read_csv('../datasets/heart-attack-prediction/heart.csv')
    # sns.set_style('whitegrid');
    # sns.pairplot(df);
    # plt.show()

    df_data = df[['age', 'sex', 'chol']]
    df_target = df[['target']]

    data = df_data.to_numpy()
    target = df_target.to_numpy()

    # split the data into train and test sets
    pivot = int(len(data) * .7)

    x_train = data[:pivot]
    y_train = target[:pivot]
    x_test = data[pivot:len(data)]
    y_test = target[pivot:len(data)]
    k = 5
    # do inference
    correct_pred=0
    for p,t in zip(x_test, y_test):
        dist_vec = [distance(d, p) for d in data]
        knn_indx = np.argsort(dist_vec)[0:k]
        knn_point = target[knn_indx]
        if stats.mode(knn_point[0])[0][0] == t:
            correct_pred = correct_pred + 1
    accuracy = correct_pred/len(x_test)
    print('Accuracy is: %f'%accuracy)


#%%
