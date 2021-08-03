# Preprocessing

Authors - Meet Sanghvi, Ashish Amresh <br>
Institution - Arizona State University <br>
Email - <Meet.Sanghvi@asu.edu>, <amresh@asu.edu>;

This module is a csv preprocesing module that can be used to process csv files and make them ready for machine learning and statistical usages

## Code Requirements
- pandas - https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html

## Usage Example
Simply import the module
> from preprocess.PREPROCESS import Preprocess

and then call it in your code

initialize preprocessing of csv by passing the file name and columns to be used

    preproc = Preprocess(config=cfg)

drop rows where field value == na
    
    preproc.dropna()

update column type e.g. convert object column to int as per mapping giving in the column data type
    
    preproc.columnDataType(data_type_for_each_column=columns_data_type) 

get final pandas data frame after preprocessing

    data = preproc.get_data() 

you can export the cleaned csv as well | pass the file name without extension

    preproc.write_csv("cleaned_data")