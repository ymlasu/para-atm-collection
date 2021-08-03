from preprocess.PREPROCESS import Preprocess
from hfm.HFM import step_aic

if __name__ == '__main__':     
    cfg = {
            "fields": ['traffic_density', 'ready_latency', 'traffic_density', 'ready_latency', 'query_latency',
                    'query_timed_out', 'sa_correct', 'interbeat_interval', 'rx', 'ry', 'rz', 'eyeblink', 'positive',
                    'neutral', 'negative', 'CLCD', 'Words_sec', 'los_severity', 'los_freq'],
            'file_name': "data.csv"
    }
    
    columns_data_type = {
            "Ss": "int",
            "condtn": "str",
            "baseline": "int",
            "highworkload": "int",
            "highworkload offnominal": "int",
            "at_sec": "int",
            "traffic_density": "int",
            "los_freq": "int",
            "los_duration_over5min": "int",
            "query": "str",
            "ready_latency": "float",
            "ready_latency_adj": "float",
            "query_latency": "float",
            "response_index": "int",
            "ready_timed_out": "int",
            "query_timed_out": "int",
            "stimuli": "str",
            "response_text": "str",
            "sa_correct": "float",
            "wl_rating": "int",
            "interbeat_interval": "float",
            "condtn_num": "int",
            "face_conf": "str",
            "rx": "float",
            "ry": "float",
            "rz": "float",
            "eyeblink": "float",
            "positive": "float",
            "neutral": "float",
            "negative": "float",
            "emo_conf": "int",
            "los_severity": "float",
            "CLCD": "int",
            "In_transmission (binary)": "int",
            "transmission started (count)": "int",
            "transmission ended (count)": "int",
            "words per transmission (syn_complexity)": "int",
            "length of transmission(in_sec)": "float",
            "Words_sec": "float",
            "time filled in previous interval (up to 5 seconds)": "float",
            "time since last transmissions": "float",
            "True Pilot or ATC": "str",
            "Interval-Pilot (p) OR ATC (a) (ap for shared intervals)": "str",
            "pilot communication time": "float",
            "air traffic communication time in previous interval": "float",
            "ratio of comms at interval P:A": "str"
    }
    
    #initialize preprocessing of csv by passing the file name and columns to be used
    preproc = Preprocess(config=cfg)
    
    # drop rows where field value == na
    preproc.dropna()
    # update column type e.g. convert object column to int as per mapping giving in the column data type
    preproc.columnDataType(data_type_for_each_column=columns_data_type) 
    
    # get final pandas data frame after preprocessing
    data = preproc.get_data() 
    
    # you can export the cleaned csv as well | pass the file name without extension
    preproc.write_csv("cleaned_data")
    
    # choose a list ofindependent variables for the step aic function
    list_of_independent_variables = ['traffic_density', 'ready_latency', 'query_latency', 'query_timed_out', 'sa_correct',
                                   'interbeat_interval', 'rx', 'ry', 'rz', 'eyeblink', 'positive', 'neutral',
                                   'negative', 'CLCD', 'Words_sec']
    
    # choose the dependent variable for the step aic function
    list_of_dependent_variables = ['los_freq'] # or los_severity
    
    model, selected_columns = step_aic(independent_variables=list_of_independent_variables,
                                       dependent_variables=list_of_dependent_variables,
                                       data=data)
    # model is of type ==> https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html
    # you can access variables within model as mentioned in this link
    
    print(model.summary())
    print(selected_columns)