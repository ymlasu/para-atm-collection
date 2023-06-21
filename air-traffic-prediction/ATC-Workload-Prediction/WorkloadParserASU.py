#!/usr/bin/env python
# coding: utf-8
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


path_to_data = ['/~/HDD1/ypang6/workload/DynamicGraph/workload_data/cpt1_all_traffic_Ss01-06.csv',
                '/~/HDD1/ypang6/workload/DynamicGraph/workload_data/cpt1_all_traffic_Ss07-10.csv',
                '/~/HDD1/ypang6/workload/DynamicGraph/workload_data/label_file.xlsx']
df1 = pd.read_csv(path_to_data[0])
df2 = pd.read_csv(path_to_data[1])
df = pd.concat([df1, df2])
dfy = pd.read_excel(path_to_data[2])
# df = df.drop(columns=['id_pair_alpha', 'dist_nm', 'dist_alt', 'los']).drop_duplicates()
# agent_ls = df['Ss'].unique()
agent_ls = [1, 2, 3, 4, 5, 6]

condition_ls = ['baseline', 'hi_wkld_nom', 'hi_wkld_off']
condition_idx = 2  # select from different workload conditions during experiments
print(
    f'start processing raw data for condition: {condition_ls[condition_idx]}')


def dist_helper(horizontal, vertical, alt):
    '''formulate to calculate the weight in the graph

    Chatterji, G. B., & Sridhar, B. (1999, June). Neural network based air traffic controller workload prediction. In Proceedings of the 1999 American Control Conference (Cat. No. 99CH36251) (Vol. 4, pp. 2620-2624). IEEE.

    horizontal: distance in NM
    vertical: distance in feet
    '''
    if alt <= 29000:
        s = 5/1000
    else:
        s = 5/2000
    return np.sqrt(horizontal**2 + s**2 * vertical**2)


# def find_mst(time, dfc, file1):
    # G = [[0]*len(dfc)]*len(dfc)
    # ids = dfc['id'].unique()
    # if ids.size<2:
    #     return
    # for idx, row in dfc.iterrows():
    #     id1, id2 = row['id_pair_alpha'].split(',')
    #     idx1, idx2 = ids.index(id1), ids.index(id2)
    #     G[idx1][idx2] = dist_helper(row['dist_nm'], row['dist_alt'], row['alt'])
    # X = csr_matrix(G)
    # Tcsr = minimum_spanning_tree(X)
    # Tcsr.toarray().astype(int)
    # return Tcsr


# change wrong response_text to, first backfill, then forward fill
dfy.loc[dfy["stimuli"] != "Rate your workload", 'response_text'] = np.NAN
dfy["response_text"] = dfy["response_text"].fillna(method='backfill')
dfy["response_text"] = dfy["response_text"].fillna(method='ffill')

# file0: nodeIDname, nodemathID=nodeRealID
# file1: nodeRealID, class
# file2: nodeRealID, time, density, lat, lon, alt
# file3: node1mathID, node2mathID
file0 = pd.DataFrame(columns=['nodeIDname', 'nodemathID'])
file1 = []
file2 = []
file3 = pd.DataFrame(columns=['txId1', 'txId2', 'dis'])
a = 0  # node index/number
T = 0

adjust_wl_dict = {1: [[1, 6, 7], [3, 7, 7], [3, 7, 7]],
                  2: [[1, 2, 4], [1, 7, 7], [1, 7, 7]],
                  3: [[1, 7, 1], [1, 7, 7], [2, 7, 7]],
                  4: [[1, 4, 1], [1, 5, 7], [1, 7, 6]],
                  5: [[2, 5, 3], [2, 7, 7], [2, 7, 7]],
                  6: [[2, 4, 2], [1, 7, 7], [7, 7, 7]],
                  }


for i in tqdm(range(1, len(agent_ls)+1)):  # agent_ls:
    df_temp = df.loc[df['Ss'] == i]
    dfy_temp = dfy.loc[dfy['Ss'] == i]

    timestamp_ls = dfy_temp['at_sec'].unique()  # df_temp['at_sec'].unique()
    try:
        assert len(timestamp_ls) == 300
    except:
        print(f'total timestamp for agent {i} is {len(timestamp_ls)}')

    # adjust_wl = np.zeros_like(timestamp_ls)
    # t1: 3min t2: 12min t3: 21min
    t1, t2, t3 = adjust_wl_dict[i][condition_idx]

    for j in timestamp_ls:
        k = condition_ls[condition_idx]


        dfy_label = dfy_temp.loc[(dfy_temp['at_sec'] == j//5*5)
                                 & (dfy_temp['condtn'] == k)]['response_text'].values
        dfy_density = dfy_temp.loc[(
            dfy_temp['at_sec'] == j//5*5) & (dfy_temp['condtn'] == k)]['traffic_density'].values
        # dfc = df_temp.loc[(df_temp['at_sec']==j) & (df_temp['condtn']==k)]
        dfc = df_temp.loc[(df_temp['at_sec']//5*5 == j)
                          & (df_temp['condtn'] == k)]

        # dfc: graph at one time step every time window = 5sec or depends on 'at_sec' available data
        # mst = find_mst(j, dfc, file1)

        # look for node
        ids = dfc['id'].unique()

        # file0: nodeIDname, nodemathID=nodeRealID
        # file 1: node real id, label
        # file 2: node real id, time, density,
        for ii in ids:
            id_information = dfc.loc[dfc['id'] == ii].iloc[0]
            file0 = file0.append(pd.DataFrame({'nodeIDname': [ii], 'nodemathID': [
                                 a]}), ignore_index=True)  # loc[len(file0.index)+1] = [i, a]

            if j <= 180:  # 0-3min: fill t1-1
                file1.append(np.asarray([a, t1-1]))  # class
            elif j > 180 and j <= 720:  # 3-12min: fill t1-1
                file1.append(np.asarray([a, t1-1]))
            elif j > 720 and j <= 1260:  # 12-21min: fill t2-1
                file1.append(np.asarray([a, t2-1]))
            else:  # 21-25min: fill t3-1
                file1.append(np.asarray([a, t3-1]))

            file2.append(np.asarray([a, j//5*5+T, dfy_density[0], id_information['lat'], id_information['lon'],
                         id_information['alt'], id_information['dist_nm'], id_information['dist_alt'], id_information['alt']]))
            a += 1

        if ids.size >= 2:
            for idx, row in dfc.iterrows():
                id1, id2 = row['id_pair_alpha'].split(',')
                if id1 in ids:
                    if id2 in ids:
                        dis = dist_helper(
                            row['dist_nm'], row['dist_alt'], row['alt'])
                        id1_mathid = file0.loc[file0['nodeIDname']
                                               == id1].iloc[-1]['nodemathID']
                        id2_mathid = file0.loc[file0['nodeIDname']
                                               == id2].iloc[-1]['nodemathID']

                        # loc[len(file3.index)+1] = [id1_mathid, id2_mathid, dis]
                        file3 = file3.append(
                            {'txId1': id1_mathid, 'txId2': id2_mathid, 'dis': dis}, ignore_index=True)

            file3 = file3.loc[file3.astype(str).drop_duplicates().index]
    T += j//5*5

file1 = pd.DataFrame(file1, columns=['txId', 'class'])
# columns=['realID', 'time','density','lat','lon','alt'])
file2 = pd.DataFrame(file2)

file0.to_csv(
    f'/~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/{condition_ls[condition_idx]}/file0_nodeMap.csv', index=False)
file1.to_csv(
    f'/~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/{condition_ls[condition_idx]}/classes.csv', index=False)
file2.to_csv(
    f'/~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/{condition_ls[condition_idx]}/features.csv', index=False)
file3.to_csv(
    f'/~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/{condition_ls[condition_idx]}/edgelist.csv', index=False)

