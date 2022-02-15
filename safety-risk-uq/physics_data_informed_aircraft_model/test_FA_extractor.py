# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:52:14 2022

@author: Abhinav
"""

import numpy as np
import pandas as pd
import datetime


# URLs for testing
#FA_URL = 'https://flightaware.com/live/flight/UAL1895/history/20200224/1225Z/KTPA/KIAH/tracklog'
#FA_URL = 'https://flightaware.com/live/flight/UAL1895/history/20220212/1620Z/KIAH/KPHX/tracklog'
#FA_URL = 'https://flightaware.com/live/flight/AAL179/history/20220209/1409Z/KJFK/KSFO/tracklog'
FA_URL='https://flightaware.com/live/flight/DAL2815/history/20220215/2050Z/KATL/KSAT/tracklog'
dfs = pd.read_html(FA_URL)
traj = dfs[0]

cols = traj.columns

# Time
col0=traj.get(cols[0])

temp1= datetime.datetime.strptime(col0[5][:15],"%a %I:%M:%S %p")
temp2= datetime.datetime.strptime(col0[6][:15],"%a %I:%M:%S %p")
duration=temp2-temp1
duration.total_seconds()
   
col01=col0[1:]
col02=col01.str.slice(0,15)

col03=pd.to_datetime(col02,format='%a %I:%M:%S %p',errors='coerce')
col04=col03-col03[1]

di = {'Mon':0, 'Tue':1, 'Wed':2, 'Thu':3, 'Fri':4, 'Sat':5, 'Sun':6}
temp=col01.str.slice(0,3)
seconds_offset = temp.replace(di)*24*60*60

col05 = col04.dt.total_seconds() + seconds_offset - seconds_offset[1]

# Latitude
col1=traj.get(cols[1])
col11=col1[1:]
col12=col11.str.extract("^(\\-?\d+\.\d{4}?)")
# Explanation of regex used for extracting latitude from the string
# ^: Start at the beginning of the string, include
# \\-?: negative sign if present, followed by
# \d+: unlimited digits in series, before
# \.: a decimal point is encountered, followed by
# \d{4}: four digits

# Longitude
col2=traj.get(cols[2])
col21=col2[1:]
col22=col21.str.extract("^(\\-?\d+\.\d{4}?)")
# See comment on regex used for extracting latitude from the string
col22.astype('float')

# Course
col3=traj.get(cols[3])
col31=col3[1:]
col32=col31.str.extract("(\\-?\d+)")
# Explanation of regex used for extracting course angle from the string
# \\-?: Include negative sign if present, followed by
# \d+: unlimited digits in series

# TAS kts
col4=traj.get(cols[4])
col41 = col4[1:]
col42=col41.str.extract("(\\-?\d+)")
# See comment on regex used for extracting course angle from the string

# TAS mph
col5=traj.get(cols[5])
col51=col5[1:]
col52=col51.str.extract("(\\-?\d+)")
# See comment on regex used for extracting course angle from the string

# Altitude ft
col6=traj.get(cols[6])
col61=col6[1:]
col61 = col61.replace(np.nan,0)
temp=pd.to_numeric(col61,errors='coerce') # The extries are presented in duplicated & concatinated form, e.g.: 475 is displayed as 475475
temp1=temp.astype(str)
len1= temp1.str.len()
col62=np.floor(temp/pow(10,len1//2)) # use pow(10,len1) to extract first half of the digits in "temp"

# ROCD
col7=traj.get(cols[7])
col71=col7[1:]
col72=col71.fillna(0) # cannot distinguish between NaN and 0 at source, we resolve this by replacing all NaNs with zeroes

names=['Time_s','Latitude','Longitude','Course_deg','TAS_kts','TAS_mph','Altitude_ft','ROCD']
traj_mat0 = pd.concat([col05,col12,col22,col32,col42,col52,col62,col72], axis=1,ignore_index = True)
traj_mat0.columns = names
traj_mat0.index = np.arange(traj_mat0.shape[0])
traj_mat0['TAS_kts']=traj_mat0['TAS_kts'].astype(float)
traj_mat0['TAS_mph']=traj_mat0['TAS_mph'].astype(float)
traj_mat0['Course_deg']=traj_mat0['Course_deg'].astype(float)

is_NaN = traj_mat0.isnull()
rows_with_NaN = is_NaN.any(axis=1) # Test for at least one NaN in each row
traj_mat = traj_mat0[~rows_with_NaN] 