#!/usr/bin/env python
# coding: utf-8
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import pdb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# path_to_data = ['/~/HDD1/ypang6/workload/DynamicGraph/workload_data/cpt1_all_traffic_Ss01-06.csv', '/~/HDD1/ypang6/workload/DynamicGraph/workload_data/cpt1_all_traffic_Ss07-10.csv','/~/HDD1/ypang6/workload/DynamicGraph/workload_data/label_file.xlsx']
# df1 = pd.read_csv(path_to_data[0])
# df2 = pd.read_csv(path_to_data[1])
# df = pd.concat([df1, df2])
# dfy = pd.read_excel(path_to_data[2])

# agent_ls = df['Ss'].unique()
# condition_ls = ['baseline', 'hi_wkld_nom', 'hi_wkld_off']
# k = condition_ls[2]

# # change wrong response_text to, first backfill, then forward fill
# dfy.loc[dfy["stimuli"] != "Rate your workload",'response_text'] = np.NAN
# dfy["response_text"] = dfy["response_text"].fillna(method='backfill')
# dfy["response_text"] = dfy["response_text"].fillna(method='ffill')

# data = pd.DataFrame(columns=['density', 'workload'])
# data['density'] = dfy.loc[dfy["condtn"] == k]['traffic_density'].values
# data['workload'] = dfy.loc[dfy["condtn"] == k]['response_text'].values


condition_ls = ['baseline', 'hi_wkld_nom', 'hi_wkld_off']
condition_idx = 2


labels = pd.read_csv(
    f'/~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/{condition_ls[condition_idx]}/classes.csv')  # txId,class
features = pd.read_csv(
    f'/~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/{condition_ls[condition_idx]}/features.csv')
features.rename(columns={'0': 'txId', '1': 'ts', '2': 'density', '3': 'lat', '4': 'lon',
                '5': 'alt', '6': 'dist_nm', '7': 'dist_alt', '8': 'alt2'}, inplace=True)
df = pd.merge(features, labels, on=['txId'])


# w/ graph features
# data = pd.DataFrame(columns=['density', 'dist_nm',
#                     'dist_alt', 'alt', 'workload'])
# data['density'] = df['density'].values
# data['workload'] = df['class'].values
# data['dist_nm'] = df['dist_nm'].values
# data['dist_alt'] = df['dist_alt'].values
# data['alt'] = df['alt'].values


# w/ density only
data = pd.DataFrame(columns=['density', 'workload'])
data['density'] = df['density'].values
data['workload'] = df['class'].values


data = np.asarray(data)
linear_regressor = LinearRegression()  # create object for the class

# perform linear regression
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False)


linear_regressor.fit(X_train, y_train)
Y_pred = np.rint(linear_regressor.predict(X_test))
# Y_pred[Y_pred < 0] = 0

# Y_pred = pd.DataFrame(Y_pred)
# labels = pd.DataFrame(data[:,1]).unsqueeze(dim=0)
num_class = 7
tpa = 0
fna = 0
fpa = 0

tp_T = np.zeros(num_class)
fn_T = np.zeros(num_class)
fp_T = np.zeros(num_class)

for cl in range(num_class):
    tp, fn, fp = 0, 0, 0
    cl_indices = (y_test == cl)
    print(cl)
    # import pdb;pdb.set_trace()
    pos = (Y_pred == cl)
    hits = Y_pred[cl_indices] == y_test[cl_indices]
    tp = hits.sum()
    # pdb.set_trace()
    tpa += tp
    fna += len(hits) - tp
    fpa += pos.sum() - tp

    tp_T[cl] = tp
    fn_T[cl] = len(hits) - tp
    fp_T[cl] = pos.sum() - tp

print('000',tpa, fna, fpa)
# pdb.set_trace()


def calculate_measure(tp, fn, fp):
    # avoid nan
    # if tp == 0:
    #     return 0, 0, 0

    p = tp * 1.0 / (tp + fp)
    r = tp * 1.0 / (tp + fn)
    if (p + r) > 0:
        f1 = 2.0 * (p * r) / (p + r)
    else:
        f1 = 0
    return p, r, f1


p, r, f1 = calculate_measure(tpa, fna, fpa)
print('precision: ', p, ' recall: ', r, ' self micro f1 score: ', f1)

p_T =[]
r_T =[]
f1_T =[]

for i in range(num_class):
    if tp_T[i] == 0 and fn_T[i] == 0:
        continue
    a1, a2, a3 = calculate_measure(tp_T[i], fn_T[i], fp_T[i])
    p_T.append(a1)
    r_T.append(a2)
    f1_T.append(a3)

macrof1 = sum(f1_T)/len(f1_T)
print('tp', tp_T, 'fp', fp_T, 'fn',fn_T,'f1',f1_T, ' self macro f1 score: ', macrof1)

aa = r2_score(y_test, Y_pred)
print('r2', aa)

microF1 = precision_score(y_test, Y_pred, average = None)#, average='micro')
macroF1 = f1_score(y_test, Y_pred, average='macro')

print(
    f'condition:{condition_ls[condition_idx]} microF1:{microF1} macroF1:{macroF1}')
