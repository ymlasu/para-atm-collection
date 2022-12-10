import zipfile as zf
import wget
import os
import scipy.io as scio
from scipy.signal import butter,decimate,filtfilt
import numpy as np
import pandas as pd
from geopy.distance import geodesic as GD
from scipy.interpolate import interp1d


class DASHlinkDownloader: 
    def __init__(self,base_url,remove_zip=True):
        self.base_url = base_url
        self.remove_zip = remove_zip
    def download_zip(self,file_name,output_directory=None):
        if output_directory is not None:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                os.chmod(output_directory,0o777)
            
        if output_directory is not None:
            if not os.path.exists(output_directory+'/'+file_name):
                fname = wget.download(self.base_url+file_name,out=output_directory)
            else:
                fname = output_directory + '/'+file_name
        else:
            if not os.path.exists(file_name):
                fname = wget.download(self.base_url+file_name)
            else:
                fname = file_name
        return fname
    def unzip(self,zipfile):
        folder = zipfile.split('.zip')[0]
        if not os.path.exists(folder):
            os.mkdir(folder)
        with zf.ZipFile(zipfile,"r") as f:
            f.extractall(folder) 
        if self.remove_zip:
            os.remove(zipfile)
        return folder
    
class DASHlinkData: 
    def __init__(self,mat_file):
        self.mat_file = mat_file
        self.data = scio.loadmat(mat_file)
        self.resampled_data_1s = None
        self.resampled_data_4s = None

    def contains_phase_no(self,phase=7):
        if phase in np.squeeze(self.data['PH'][0][0][0]):
            return True
        else:
            return False
        
    def temporal_resample_to_1_second(self,key):
        if self.resampled_data_1s is None:
           self.resampled_data_1s = pd.DataFrame() 
        tem_rate=int(self.data[key][0][0][1])
        if tem_rate!=0:
            tem_list=[float(i) for i in self.data[key][0][0][0]]
            if tem_rate==1:
                new_tem=tem_list
            if tem_rate<1:
                new_tem=np.repeat(tem_list,int(1/tem_rate))
            if tem_rate>1:
                new_tem=tem_list[ : :tem_rate]
        self.resampled_data_1s[key]=new_tem
    
    def temporal_resample_to_4_seconds(self,key):
        if self.resampled_data_4s is None:
            self.resampled_data_4s = pd.DataFrame() 

        tem_data=np.squeeze(self.data[key][0][0][0]).astype(float) 
        tem_rate=int(self.data[key][0][0][1])
        if tem_rate!=0:
            if key != 'MSQT_1':
                b,a=butter(3,0.5)
                ### b and a controls the filtering, when a is smaller, the filterig is less severe.
                filtered_data=filtfilt(b,a,tem_data)
                if tem_rate>4:
                    down_sample_data=decimate(filtered_data,int(tem_rate//4),8,zero_phase=True)
                    #print(down_sample_data)
                    sampled_data=down_sample_data
                if tem_rate==4:
                    sampled_data=filtered_data
                if tem_rate<4:
                    sampled_data= np.repeat(tem_data,int(4//tem_rate))
            if key == 'MSQT_1':
                sampled_data=np.repeat(tem_data,int(4//tem_rate))
        self.resampled_data_4s[key]=sampled_data
       
    def lands_at_airport(self,airport_lat_lon=[44.88526995556498, -93.2015923365669],key_list=['LATP','LONP','MSQT_1']):
        for key in key_list:
            self.temporal_resample_to_1_second(key)
        self.resampled_data_1s.dropna(axis=0,subset=key_list,inplace=True)
        if np.any(self.resampled_data_1s['MSQT_1']==1.0):
            td_index = self.get_touchdown_index()
            tem_lat = self.resampled_data_1s['LATP'][td_index]
            tem_lon = self.resampled_data_1s['LONP'][td_index]
            sqt_switch = self.resampled_data_1s['MSQT_1'][td_index]
            pos=(tem_lat,tem_lon)
            dis=GD(airport_lat_lon,pos).miles
            if (dis<2) and (sqt_switch==1.0):
                return True
            else:
                return False
        else:
            return False

### obtain the touchdown index
    def get_touchdown_index(self,which='1s'):
        if which =='1s':
            if self.resampled_data_1s is None:
                self.temporal_resample_to_1_second('MSQT_1')
            df = self.resampled_data_1s
        elif which=='4s':
            if self.resampled_data_4s is None:
                self.temporal_resample_to_4_seconds('MSQT_1')
            df = self.resampled_data_4s
        idx = df[df['MSQT_1']==1.0].index
        if idx is not None:
            td_index = np.max(idx)
        else:
            td_index=0
        return td_index
    
    def get_data_at_heights_in_ft(self,heights,step=4,which='4s'):
        if which=='4s':
            df = self.resampled_data_4s
        elif which=='1s':
            df = self.resampled_data_1s
        
        td_index = self.get_touchdown_index(which=which)
        td_index = td_index-step
        idxes=np.arange(td_index-np.min([1000,td_index]),td_index+1)
        td_alt = df.BAL1.values[td_index]
        td_altr = df.ALTR.values[td_index]
        td_lat=df['LATP'].values[td_index]
        td_lon=df['LONP'].values[td_index]

        alts = td_alt + heights

        df_new = pd.DataFrame()
        old_heights=df.BAL1.loc[idxes]
        for col in df.columns:
            if col != "BAL1":
                old_vals=df[col].loc[idxes]
                f =interp1d(old_heights,old_vals)
                new_vals=f(alts)
                df_new[col]=new_vals
            else:
                df_new[col]=alts
                
        df_new['TD_ALTR']=td_altr
        df_new['TD_LAT']=td_lat
        df_new['TD_LON']=td_lon
        df_new['TD_ALT']=td_alt
        df_new['heights']=heights
        
        dists = np.array([GD([lat,lon],[td_lat,td_lon]).miles*1609.34 for lat,lon in zip(df_new.LATP.values,df_new.LONP.values)])
        df_new['DIST']=dists
        
        return df_new
            
        
        
        
