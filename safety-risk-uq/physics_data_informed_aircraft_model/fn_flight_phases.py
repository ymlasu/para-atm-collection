# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:39:00 2021

@author: Abhinav
"""

import numpy as np
from sklearn.cluster import KMeans

def pyfn_flight_phase(ROCD, Altitude_ft, Time_s):
    
    # 1. detect multiple altitude levels during cruise
    ROCD_tol = 50
    ROCD_indicator_neutral = np.where(abs(ROCD)<ROCD_tol)[0]
    
    possible_altitude_levels_1 = Altitude_ft[ROCD_indicator_neutral]
    Ix_possible_altitude_levels_2 = np.where(possible_altitude_levels_1>0.5*max(possible_altitude_levels_1))
    
    temp=possible_altitude_levels_1[Ix_possible_altitude_levels_2]
    temp2=temp.reshape(-1, 1)
    
    sse=[]
    for kmeans_k in range(1,5):
        kmeans = KMeans(
            init='k-means++',
            n_clusters=kmeans_k,
            n_init=10,
            max_iter=300,
        )
        kmeans.fit(temp2)
        sse = np.append(sse,kmeans.inertia_)
    if sse.size==np.count_nonzero(sse):
        # find elbow
        rel_diff = np.divide(sse[1:]-sse[:-1],sse[:-1])
        n_clusters_cruise = 1 + np.argmin(rel_diff)
    else: 
        first_zero_in_sse = np.where(sse == 0)[0][0]
        n_clusters_cruise = first_zero_in_sse
    
    kmeans = KMeans(
        init='k-means++',
        n_clusters=n_clusters_cruise,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(temp2)
    kmeans.cluster_centers_[:,0]
    
    time_cluster_all = []
    alt_cluster_all = []
    for ix in range(n_clusters_cruise):
        indices_cluster_1 = np.where(kmeans.labels_==ix)[0]
        time_cluster_all.append(Time_s[ROCD_indicator_neutral[Ix_possible_altitude_levels_2[0][indices_cluster_1]]])
        alt_cluster_all.append(Altitude_ft[ROCD_indicator_neutral[Ix_possible_altitude_levels_2[0][indices_cluster_1]]])
    
    StartOfCruise = min(ROCD_indicator_neutral[Ix_possible_altitude_levels_2])
    EndOfCruise = max(ROCD_indicator_neutral[Ix_possible_altitude_levels_2])
    
    # 2. Detect Top of Climb and Top of Descent
    Ix_ToC = StartOfCruise-1 # index of Top of Climb
    Ix_ToD = EndOfCruise+1 # index of Top of Descent
    
    # 3. Formatting descent phase
    ROCD_descent_phase = ROCD[Ix_ToD:]
    ROCD_neutral_descent = np.where(abs(ROCD_descent_phase)<ROCD_tol)[0]
    if ROCD_neutral_descent.size==0:
        hold_descent_altitudes = []
    else: 
        Alt_neutral_descent = Altitude_ft[EndOfCruise+ROCD_neutral_descent]
        
        temp3=Alt_neutral_descent.reshape(-1, 1)
        
        sse=[]
        for kmeans_k in range(1,10):
            kmeans = KMeans(
                init='k-means++',
                n_clusters=kmeans_k,
                n_init=10,
                max_iter=300,
            )
            kmeans.fit(temp3)
            sse = np.append(sse,kmeans.inertia_)
        
        if sse.size==np.count_nonzero(sse):
            # find elbow
            rel_diff = np.divide(sse[1:]-sse[:-1],sse[:-1])
            n_clusters_descent = 1 + np.argmin(rel_diff)
        else: 
            first_zero_in_sse = np.where(sse == 0)[0][0]
            n_clusters_descent = first_zero_in_sse
        
        verify_length = []
        for ix in range(1,first_zero_in_sse):
            kmeans = KMeans(
                init='k-means++',
                n_clusters=ix,
                n_init=10,
                max_iter=300,
            )
            kmeans.fit(temp3)
        
            clusters_with_adequte_length = 0
            for jx in range(n_clusters_descent):
                indices_cluster_1 = np.where(kmeans.labels_==jx)[0]
                clusters_with_adequte_length = clusters_with_adequte_length+(indices_cluster_1.size>5)
            verify_length.append(clusters_with_adequte_length==ix)
        if sum(verify_length)==0:
            valid_number_of_clusters = 0
            hold_descent_altitudes = []
        else:
            valid_number_of_clusters = 1 + max(np.where(verify_length)[0])
            
            kmeans = KMeans(
                init='k-means++',
                n_clusters=valid_number_of_clusters,
                n_init=10,
                max_iter=300,
            )
            kmeans.fit(temp3)
            hold_descent_altitudes = kmeans.cluster_centers_[:,0]


    return Ix_ToC, Ix_ToD, time_cluster_all, alt_cluster_all, hold_descent_altitudes