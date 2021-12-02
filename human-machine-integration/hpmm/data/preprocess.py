import pandas as pd

def dataset(data):
    # according to experiment design, remove redundant variables
    data = data
    data_domain = data.drop(
        columns=['Ss', 'condtn', 'los_dur_over5min', 'query_timed_out', 'ready_timed_out', 'ready_latency_adj',
                 'cum_los_dur',
                 'stimuli', 'response_text', 'condtn_num'])
    # fill in missing values
    data_domain['sa_correct'].fillna(data_domain['sa_correct'].mode()[0], inplace=True)
    data_domain.fillna(data_domain.mean(), inplace=True)

    # transform categorical data
    data_domain = data_domain.join(pd.get_dummies(data_domain.sa_correct))
    data_domain = data_domain.join(pd.get_dummies(data_domain['query']))
    data_domain = data_domain.drop(columns=['query', 'sa_correct'])
    return data_domain