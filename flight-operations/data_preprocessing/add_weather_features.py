import pandas as pd
import numpy as np
import glob, os

if __name__ == "__main__":
    dir = '/media/ypang6/Yutian Pang 4TB/Research/Peng_Data/data_preprocessing'

    weather_dir = dir + '/Weather_KATL_20190801.csv'
    df_w = pd.read_csv(weather_dir)
    df_w = df_w[['windspeed', 'winddir', 'cloudcover', 'visibility', 'humidity']]
    df_w['round-60'] = df_w.index

    file_dirs = glob.glob(dir + '/processed_features/*.csv')
    for file_dir in file_dirs:
        df = pd.read_csv(file_dir)
        df = df.merge(df_w, on=['round-60'])
        df.to_csv(dir+'/processed_features_weather/'+file_dir.split('/')[-1])
        del df

    
