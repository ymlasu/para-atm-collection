from data.preprocess import dataset
from model.catboost_train import train
from util.fea_im import fea_im
import pandas as pd

if __name__ == '__main__':
    # import data source
    data_path = './data/human_data.csv'
    data = pd.read_csv(data_path)

    # data precess
    human_data = dataset(data)

    # CatBoost model training
    model = train(human_data)

    # CatBoost model importance
    fea_im = fea_im(model, human_data)




