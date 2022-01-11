# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:39:57 2021

@author: Abhinav
"""


import numpy as np
import pandas as pd
import datetime

def pyfn_FlightAware_trajectory_extractor(FA_URL):   
    dfs = pd.read_html(FA_URL)
    traj = dfs[0]
    
    cols = traj.columns
    col0=traj.get(cols[0])
    #Ix_dep_1 = tmp1.str.contains("Departure")
    #Ix_arr_1 = tmp1.str.contains("Arrival")
    
    #Ix_dep=[i for i, x in enumerate(Ix_dep_1) if x]
    #Ix_arr=[i for i, x in enumerate(Ix_arr_1) if x]
    
    temp1= datetime.datetime.strptime(col0[5][:15],"%a %I:%M:%S %p")
    temp2= datetime.datetime.strptime(col0[6][:15],"%a %I:%M:%S %p")
    duration=temp2-temp1
    duration.total_seconds()
    
    #col01=col0[(Ix_dep[0]+1):(Ix_arr[0]-1)]
    col01=col0[1:]
    col02=col01.str.slice(0,15)
    
    col03=pd.to_datetime(col02,format='%a %I:%M:%S %p')
    #col04=col03-col03[Ix_dep[0]+1]
    col04=col03-col03[1]
    
    di = {'Mon':0, 'Tue':1, 'Wed':2, 'Thu':3, 'Fri':4, 'Sat':5, 'Sun':6}
    temp=col01.str.slice(0,3)
    seconds_offset = temp.replace(di)*24*60*60
    
    #col05 = col04.dt.total_seconds() + seconds_offset - seconds_offset[Ix_dep[0]+1]
    col05 = col04.dt.total_seconds() + seconds_offset - seconds_offset[1]
    
    # Latitude
    col1=traj.get(cols[1])
    #col11=col1[(Ix_dep[0]+1):(Ix_arr[0]-1)]
    col11=col1[1:]
    col12=col11.str.extract("^(\\-?\d+\.\d{4}?)") # regex extracts the following: from start opf string (^) with possible -ve (\\-?) unlimited digits in series (\d+) a decimal point (\.) four digits \d{4} 
    
    # Longitude
    col2=traj.get(cols[2])
    #col21=col2[(Ix_dep[0]+1):(Ix_arr[0]-1)]
    col21=col2[1:]
    col22=col21.str.extract("^(\\-?\d+\.\d{4}?)")
    col22.astype('float')
    
    # Course
    col3=traj.get(cols[3])
    #col31=col3[(Ix_dep[0]+1):(Ix_arr[0]-1)]
    col31=col3[1:]
    col32=col31.str.extract("(\\-?\d+)")
    
    # TAS kts
    col4=traj.get(cols[4])
    #col41=col4[(Ix_dep[0]+1):(Ix_arr[0]-1)]
    col41 = col4[1:]
    col42=col41.str.extract("(\\-?\d+)")
    
    # TAS mph
    col5=traj.get(cols[5])
    #col51=col5[(Ix_dep[0]+1):(Ix_arr[0]-1)]
    col51=col5[1:]
    col52=col51.str.extract("(\\-?\d+)")
    
    # Altitude ft
    col6=traj.get(cols[6])
    #col61=col6[(Ix_dep[0]+1):(Ix_arr[0]-1)]
    col61=col6[1:]
    col61 = col61.replace(np.nan,0)
    temp=col61.astype(np.int64)
    temp1=temp.astype(str)
    len1= temp1.str.len()
    col62=np.floor(temp/pow(10,len1//2))
    #use pow(10,len1) to extract first digit, and so on, till half of len1
    
    # ROCD
    col7=traj.get(cols[7])
    #col71=col7[(Ix_dep[0]+1):(Ix_arr[0]-1)]
    col71=col7[1:]
    col72=col71.fillna(0)
    
    names=['Time_s','Latitude','Longitude','Course_deg','TAS_kts','TAS_mph','Altitude_ft','ROCD']
    traj_mat = pd.concat([col05,col12,col22,col32,col42,col52,col62,col72], axis=1,ignore_index = True)
    traj_mat.columns = names
    traj_mat.index = np.arange(traj_mat.shape[0])
    
    Time_s = np.array(col05)
    Latitude = np.array(col12)
    Longitude = np.array(col22)
    Course_deg = np.array(col32)
    TAS_kts = np.array(col42)
    TAS_mph = np.array(col52)
    Altitude_ft = np.array(col62)
    ROCD = np.array(col72)
    
    return traj_mat, Time_s, Latitude, Longitude, Course_deg, TAS_kts, TAS_mph, Altitude_ft, ROCD;