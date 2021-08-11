# Human Factors Module (HFM)

Authors - Mustafa Demir, Nancy Cooke, Sarah Ligda, Chris  Lieber, Meet Sanghvi, Ashish Amresh <br>
Institution - Arizona State University <br>
Email - <mustafa.demir@asu.edu>, <Nancy.Cooke@asu.edu>, <sligda@asu.edu>, <clieber@asu.edu>, <Meet.Sanghvi@asu.edu>, <amresh@asu.edu>;

This module performs a Step Wise AIC to find the best formula out of all the independent variables. 

## Args:
    independent_variables (list): independent variables
    dependent_variables (list): dependent variables
    kwargs: extra keyword argments for model (e.g., data, family)

## Returns:
    model: a model that seems to have the smallest AIC
    selected: a list of the columns selected as best fields

HFM (this module) is a purely statistical model

## Code Requirements
- pandas - https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
- numpy - https://numpy.org/install/
- statsmodels - https://www.statsmodels.org/stable/install.html

## How to use this file? check main.ipynb file for reference
- Edit the fields and file name to be used from the csv data file you are passing
- Edit the mapping dictionary = data_type_for_each_column ==> which state which column from the csv file is what type of data
- Edit the list of independent variables

## Example
Simply import the module
> from hfm.HFM import step_aic

and then call it in your code
>     model, selected_columns = step_aic(independent_variables=list_of_independent_variables, dependent_variables=list_of_dependent_variables,data=data)

model is of type ==> https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html & you can access variables within model as mentioned in this link
