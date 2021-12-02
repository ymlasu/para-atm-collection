import matplotlib.pyplot as plt

def fea_im(model, data):
    fea_ = model.feature_importances_
    # fea_name = model.feature_names_
    X_name = data.drop(columns=['los_freq'])
    fea_name = list(X_name.columns)
    fea_name = [str(j) for j in fea_name]

    plt.figure(figsize=(16, 9))
    plt.title('CatBoost feature importance')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.barh(fea_name,fea_,height =0.5)
    plt.show()