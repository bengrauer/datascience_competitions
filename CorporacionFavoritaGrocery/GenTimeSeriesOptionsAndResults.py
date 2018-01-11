#=====================================================================================================
# This script is the main work-horse.  It will take input parameters and then perform a
#  hyperparameter serach on the different settings for time-series.  The process generates two files:
#  1) The output results of all the different parameter combinations.
#  2) The results of the best item.  Originally started with lowest AIC number.
#=====================================================================================================

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numba

import time
import datetime

import platform
import logging
import sys
import os
from os import path
# works across all three OS
from sys import platform

import math
import string
import gc

import warnings
import itertools

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

#import seaborn as sns
#import matplotlib.pyplot as plt


# A global parameter - which controls which file we use.
# There is a smaller file vs a fully imputed version.  The smaller file creates the series in smaller chunks
# If set to TRUE - hovers around 3.5 GB.  Without - load starts at about 7 GB
# For the mac - I need smaller chunks.  For the big machine - I thought I could have saved extra cpu cycles with a
# fully imputed version.  That ended up not being the case, so I ended up with the smaller file.
CONST_USE_SMALLER_MEMORY_FILE = True

# function to display time of actions
def print_elapsed_time(statement, elapsed_time, optSuppressPrint = 0):
    if elapsed_time > 60:
        timeElapsed = statement + ' : ' + str(int(round(elapsed_time / 60))) + ' min and ' + str(int(round(elapsed_time / (60 * 2)) )) + ' seconds'

        if optSuppressPrint == 0:
            print(timeElapsed)
        return (timeElapsed)
    else:
        timeElapsed = statement + ' : ' + str(int(round(elapsed_time))) + ' seconds'
        if optSuppressPrint == 0:
            print(timeElapsed)
        return (timeElapsed)
        # Determine if the series is stationary or not

# Store in a function for later use!
def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']

    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))

    if result[1] <= 0.05:
        print(
            "strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

# Concatenating the inputs of data to prepare to pass into functions
# originally in the muti-threaded model I had to merge them into an array of args as I couldn't pass in multiple inputs
def fn_concat_args(input_store_nbr, input_item_nbr, input_ts_order, input_ts_seasonal_order, input_start_predict):
    '''
        temp_store_nbr = 1
        temp_item_nbr = 96995
        temp_ts_order=(1,1,0)
        temp_ts_seasonal_order=(0,0,0,14)
        temp_start_predict = 300
    '''
    args = [[input_store_nbr, input_item_nbr, input_ts_order, input_ts_seasonal_order, input_start_predict]]
    return args

# Check if the file exists
def fn_determine_file_exists(fileName):
    fileExists = False

    if path.isfile(fileName):
        fileExists = True

    return  fileExists

# Will determine where the file / process left off to pick back up
def fn_determine_file_last_run(fileName):

    # Read In
    df_leftOff = pd.read_csv(fileName)

    # df_test_iteration = pd.DataFrame(np.unique(df_test[['store_nbr', 'item_nbr']], axis=0),columns=('store_nbr', 'item_nbr'))

    # Grab the unique store number + item number
    df_leftOff = pd.DataFrame(np.unique(df_leftOff[['store_nbr','item_nbr']], axis=0), columns=('store_nbr','item_nbr'))
    df_leftOff.sort_values(['store_nbr','item_nbr'], ascending=[True, True], inplace=True)

    # Set a Processed Flag for all the entries
    df_leftOff['processed'] = 1

    # Return a data frame to join later
    return df_leftOff

# multi-thread - ended up abandoning this architecture for the multi-os process wrapper
def fn_multithread_TS_param_config(func, args, workers):
    begin_time = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func, args, [begin_time for i in range(len(args))])
        # Log logger.info('Total Items to Process: ' + str(len(fullArr)))
    return res

# Function to loop through each of the time series / seasonality parameters and call into a ts function
# that tries the configuration and returns the output
def fn_loop_timeseries_param_options(args, base):

    # split up the args
    arg_storeNumber = args[0]
    arg_itemNumber = args[1]

    # turn off the warnings
    warnings.filterwarnings("ignore")

    # Log
    logger.info('Start Processing Item: ' + str(arg_itemNumber))

    # print('Total Possibilities: ' + str(len(param_pdq) * len(param_seasonal_pdq)))
    start_time = time.time()

    tempCounterImplemented = 0  # for yes / 0 for no
    tempCounterMax = 60
    tempCounter = 0

    # Print the number of possibilities
    i = 0
    j = 0

    CONST_TS_START = 300

    for i in range(0, len(param_pdq)):
        for j in range(0, len(param_seasonal_pdq)):

            '''
            if (tempCounter != 0 and tempCounter % 500 == 0):
            #if i%2==0:
                sLogMessage = 'Processing Store: [' + str(arg_storeNumber) + ':' + str(arg_itemNumber) + '] index is: ' + str(
                    tempCounter) + ' out of ' + str((len(param_pdq) * len(param_seasonal_pdq)))
                print(sLogMessage)

                logger.info(sLogMessage)
            '''

            # Here I am running the parameter search on each item
            if tempCounter == 0:

                parmDF = fn_attempt_timeseries_param_config(
                    fn_concat_args(arg_storeNumber, arg_itemNumber, param_pdq[i], param_seasonal_pdq[j], CONST_TS_START)[0])

            else:
                parmDF = parmDF.append(fn_attempt_timeseries_param_config(
                    fn_concat_args(arg_storeNumber, arg_itemNumber, param_pdq[i], param_seasonal_pdq[j], CONST_TS_START)[0]))
                a = 1

            # increment temp counter
            tempCounter = tempCounter + 1

            # Print every 300 combinations

            # INNER LOOP
            # if we are implementing a temporary counter, and max is reached
            if tempCounterImplemented == 1:
                if tempCounter >= tempCounterMax:
                    break

        # OUTER LOOP
        # if we are implementing a temporary counter, and max is reached
        if tempCounterImplemented == 1:
            if tempCounter >= tempCounterMax:
                break

    # Log
    logger.info('Stop Processing Item: ' + str(arg_itemNumber) + ' - ' + print_elapsed_time('Time Finished: ', time.time() - start_time, 1))

    return parmDF

    # print(tempCounter)
    # print_elapsed_time('Time Finished: ', time.time() - start_time)

# This function will attempt to run the time series based on the parameters being passed in.
def fn_attempt_timeseries_param_config(args):
    warnings.filterwarnings("ignore")  # specify to ignore warning messages

    # Constants for data existing or not
    CONST_DATA_EXISTS = 'YES'
    CONST_NO_DATA_EXISTS = 'NO'
    CONST_ERROR_EXISTS = 'ERROR'

    # Need to split up the args
    # ensure prefixed with arg_ to avoid other test items
    arg_storeNumber = args[0]
    arg_itemNumber = args[1]
    arg_ts_order = args[2]
    arg_ts_seasonal_order = args[3]
    arg_start_predict = args[4]

    # filter the TS criteria
    dfTS = df_train[(df_train['store_nbr'] == arg_storeNumber) & (df_train['item_nbr'] == arg_itemNumber)].copy()

    # Set a default list in case the TS errors
    param_diag_list = [arg_storeNumber, arg_itemNumber, CONST_ERROR_EXISTS, arg_ts_order, arg_ts_seasonal_order, 0, 0,
                       0, 0, 0]

    # If there is no data to plot then exit
    if len(dfTS) > 0:


        # Imput the missing dates going all the way back
        if CONST_USE_SMALLER_MEMORY_FILE == True:
            dfTS = pd.concat([dfTS, idx], axis=1)
            dfTS = dfTS.asfreq('D')

        # When we joined to dates, now replace everything else for the single unit we are working on
        dfTS['id'].replace({np.nan: 0}, inplace=True)
        dfTS['item_nbr'].replace({np.nan: arg_itemNumber}, inplace=True)
        dfTS['store_nbr'].replace({np.nan: arg_storeNumber}, inplace=True)
        dfTS['unit_sales'].replace({np.nan: 0}, inplace=True)
        dfTS['onpromotion'].replace({np.nan: False}, inplace=True)

        sales_cycle, sales_trend = sm.tsa.filters.hpfilter(dfTS.unit_sales)
        dfTS['trend'] = sales_trend
        dfTS['cycle'] = sales_cycle

        # Just the cycle
        # OR sm.tsa.filters.hpfilter(dfTS.unit_sales)[1]

        # if we hit an error trying to perform a TS analysis, then skip over it in the paramater seraching
        try:

            # the forecasting
            model = sm.tsa.statespace.SARIMAX(dfTS['unit_sales'], order=arg_ts_order,
                                              seasonal_order=arg_ts_seasonal_order,
                                              enforce_stationarity=False, enforce_invertibility=False)
            # model = sm.tsa.statespace.SARIMAX(dfTS['unit_sales'], order=(1,2,0), seasonal_order=(1,1,1,7))

            # conrol display results - add memory to a mac run for some reason
            #results = model.fit()
            results = model.fit(disp=0)

            # for 2013-01-01 to current.  Use 1600 / 2150
            # for 2016-08-01 to current.  Use 0 / 396
            # for 2016-08-01 to 2017-08-15 Use 380
            # for 2017-06-01 to 2017-08-30 - start 300 / end - 396
            # for 2017-06-01 to 2017-08-15 - start 300 / end - 381
            dfTS['forecast'] = results.predict(start=arg_start_predict, end=396)  # , dynamic=True

            # Drop the last entry as I went one too far out
            dfTS.drop(dfTS.index[len(dfTS) - 1], inplace=True)

            dfTS['cycle_rnd'] = dfTS['cycle'].apply(math.ceil)
            dfTS['forecast_rnd'] = dfTS['forecast']
            dfTS['forecast_rnd'].replace({np.nan: 0}, inplace=True)
            dfTS['forecast_rnd'] = dfTS['forecast_rnd'].apply(math.ceil)
            # dfTS['forecast_sq'] = dfTS['forecast_rnd'] * dfTS['forecast_rnd']

            # Configure MSE
            # grabbing dates from 2017-06-01 to 2017-08-15, which lines up close to a start date of 300
            y = dfTS['unit_sales']["2017-06-01":"2017-08-15"]
            y_frcast = dfTS['forecast']["2017-06-01":"2017-08-15"]
            y_frcast_rnd = dfTS['forecast_rnd']["2017-06-01":"2017-08-15"]

            mse_frcast = ((y_frcast - y) ** 2).mean()
            mse_frcast_rnd = ((y_frcast_rnd - y) ** 2).mean()

            sse_frcast = ((y_frcast - y) ** 2).sum()
            sse_frcast_rnd = ((y_frcast_rnd - y) ** 2).sum()

            # Here we need to record the following:
            # 1) store_nbr
            # 2) item_nbr
            # 3) full param set
            # 4) AIC from the model
            #    SME from the days of Aug 01 - Aug 14.
            # 5) SME from regular forecast
            # 6) SME from forecast rounded to nearest whole unit
            # print(mse_frcast)
            # print(mse_frcast_rnd)
            # print(results.aic)
            # print(results.model.order)
            # print(results.model.seasonal_order)
            # print(results.model.seasonal_periods)

            # Set the parameter list
            param_diag_list = [arg_storeNumber, arg_itemNumber, CONST_DATA_EXISTS, arg_ts_order, arg_ts_seasonal_order,
                               results.aic, mse_frcast, mse_frcast_rnd, sse_frcast, sse_frcast_rnd]

        # If we hit an error, continue on
        except:
            pass

    # ELSE
    else:
        # print('nothing')
        # set a blank diag list with a constant of no data existing
        param_diag_list = [arg_storeNumber, arg_itemNumber, CONST_NO_DATA_EXISTS, arg_ts_order, arg_ts_seasonal_order,
                           0, 0, 0, 0, 0]

    # END IF STATEMENT

    if optionalPlotFunctionTS == 1:
        dfTS[['unit_sales', 'forecast', 'forecast_rnd']]["2017-06-01":].plot(figsize=(12, 8))

    # Turn the list into a data frame
    colNames = (
    'store_nbr', 'item_nbr', 'data_present', 'model_order', 'model_seasonal_order', 'model_aic', 'mse_frcst',
    'mse_frcst_rnd', 'sse_frcst', 'sse_frcst_rnd')
    df_param_diag = pd.DataFrame(param_diag_list).T
    df_param_diag.columns = colNames

    # return the data frame
    return df_param_diag

# This function will actually run the time series forecast selected based on the hyperparameter search above
# def config_time_Series(storeNumber, itemNumber, ts_order, ts_seasonal_order, start_predict):
def fn_config_timeseries(args):

    # Need to split up the args
    # ensure prefixed with arg_ to avoid other test items
    arg_storeNumber = args[0]
    arg_itemNumber = args[1]
    arg_ts_order = args[2]
    arg_ts_seasonal_order = args[3]
    arg_start_predict = args[4]

    # filter the TS criteria
    dfTS = df_train[(df_train['store_nbr'] == arg_storeNumber) & (df_train['item_nbr'] == arg_itemNumber)].copy()

    # If there is no data to plot then exit
    if len(dfTS) > 0:

        # If we are using a smaller memory cycle
        if CONST_USE_SMALLER_MEMORY_FILE == True:

            # Imput the missing dates going all the way back
            dfTS = pd.concat([dfTS, idx], axis=1)
            dfTS = dfTS.asfreq('D')

            # When we joined to dates, now replace everything else for the single unit we are working on
            dfTS['id'].replace({np.nan: 0}, inplace=True)
            dfTS['item_nbr'].replace({np.nan: arg_itemNumber}, inplace=True)
            dfTS['store_nbr'].replace({np.nan: arg_storeNumber}, inplace=True)
            dfTS['unit_sales'].replace({np.nan: 0}, inplace=True)
            dfTS['onpromotion'].replace({np.nan: False}, inplace=True)

        # the forecasting
        model = sm.tsa.statespace.SARIMAX(dfTS['unit_sales'], order=arg_ts_order, seasonal_order=arg_ts_seasonal_order,
                                          enforce_stationarity = False, enforce_invertibility = False)

        # model = sm.tsa.statespace.SARIMAX(dfTS['unit_sales'], order=(1,2,0), seasonal_order=(1,1,1,7))

        #results = model.fit()
        results = model.fit(disp=0)

        # for 2013-01-01 to current.  Use 1600 / 2150
        # for 2016-08-01 to current.  Use 0 / 396
        # for 2016-08-01 to 2017-08-15 Use 380
        # for 2017-06-01 to 2017-08-30 - start 300 / end - 396
        # for 2017-06-01 to 2017-08-15 - start 300 / end - 381
        dfTS['forecast'] = results.predict(start=arg_start_predict, end=396)  # , dynamic=True

        # rounded forecast
        dfTS['forecast_rnd'] = dfTS['forecast']
        dfTS['forecast_rnd'].replace({np.nan: 0}, inplace=True)
        dfTS['forecast_rnd'] = dfTS['forecast_rnd'].apply(math.ceil)

        # plot - need to make this optional
        if optionalPlotFunctionTS == 1:
            dfTS[['unit_sales', 'forecast_rnd']]["2017-06-01":].plot(figsize=(12, 8))
        else:
            # If we are not plotting, then only include the month of August (15 - 30 if the submission)
            # We will keep the first 15 days, just in case
            dfTS = dfTS[:]["2017-08-01":]

        return dfTS

    else:
        # print('No Data: Store: ' + str(storeNumber) + ' - ' + str(itemNumber) + '\n')
        return dfTS

    # clean up
    # dfTS = null (This  will not return memory back to OS in python)



if __name__ == "__main__":
# def main(arg1, arg2, arg3, arg4, arg5, arg6):
#def main(arg1, arg2, arg3):

    # specify to ignore warning messages
    # I do not encourage this, yet I only had 3 months to get this up and running/submitted.
    #  in addition I think it added memory to the shell in mac?
    warnings.filterwarnings("ignore")

    # ===========================
    # SET SCRIPT INPUT PARAMETERS
    # ===========================

    # Mandatory Store Number
    file_args_store_nbr = 0
    file_args_store_nbr = int(sys.argv[1])

    # (Optional) System/Directory
    #file_args_systemDirectory = ''
    #if len(sys.argv) >= n:
    #    file_args_systemDirectory = sys.argv[n]
    #    if (file_args_systemDirectory) != '':
    #        print('we will set directory from here - future use of AWS')

    # (Optional) Wipe Existing Export Files (to start over
    # True or False.  Default is False
    file_args_ForceWipeExistingFiles = False
    if len(sys.argv) >= 3:
        file_args_ForceWipeExistingFiles = sys.argv[2]

    # (Optional) Test Number of Records
    file_args_num_test_rec = 0
    if len(sys.argv) >= 4:
        file_args_num_test_rec = int(sys.argv[3])

    # (Optional) - Num of Threads (switched back to single-thread / multi-process for file appending,
    #      the multi-threading did not have the best efficiencies)
    file_args_num_threads = 1
    #if len(sys.argv) >= n:
    #    file_args_num_threads = int(sys.argv[n])

    # (Optional) Item Number
    file_args_test_item_nbr = 0
    if len(sys.argv) >= 5:
        file_args_test_item_nbr = int(sys.argv[4])

    print('Store Number ' + str(file_args_store_nbr))

    printStoreNum = 'Store Num: ' + str(file_args_store_nbr) + ' - '

    if file_args_num_test_rec >= 0:
        print(printStoreNum + 'Test Records: ' + str(file_args_num_test_rec))


    # Default parameters
    # An option to plot - used in prior scripts
    optionalPlotFunctionTS = 0
    # Variable to specify if we include a header one time in the files
    includeHeaderRunOnce = True
    # Determine if we are resuming a previous file
    resumeRunningPreviousFile = False

    # ===========================
    # DETERMINE DIRECTORY + FILES
    # ===========================
    # Determine the OS as mac / win  aws will have different directories
    # set directory and files
    # If we have windows
    # if platform.system() == 'Windows':
    if platform == 'win32':
        directory = 'D:\\project\\data\\kg_corpgroc\\'
        exportDirectory = directory + 'export\\'

    # Mac
    #elif platform.system() == 'Darwin':
    elif platform == 'darwin':
        directory = '//Project/data/kg_corpgroc/'
        exportDirectory = directory + 'export/'

    # AWS
    elif platform == 'linux':
        directory = '//data/'
        exportDirectory = directory + 'export/'

    exportParamOptionsFileName = exportDirectory + 'export_param_' + str(file_args_store_nbr) + '.csv'
    exportResultsSubmissionFileName = exportDirectory + 'export_results_' + str(file_args_store_nbr) + '.csv'

    exportLogName = exportDirectory + 'export_log_' + str(file_args_store_nbr) + '.log'


    # ===========================
    # SETUP LOGGING
    # ===========================
    # Wipe any existing log file
    if path.isfile(exportLogName):
        os.remove(exportLogName)

    # We may set another parameter to pass in to wipe the existing param options and results submissions

    # SEt logging information
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(exportLogName)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info('Start Logging')



    # ================
    # LOAD THE DATA
    # ================
    print(printStoreNum + 'Start File Read')
    
    file_test = 'test.csv'
    #file_train = 'train.csv'
    if CONST_USE_SMALLER_MEMORY_FILE == True:
        file_train = 'train_20160801_20170814.csv'
    else:
        file_train = 'train_20160801_20170901_imputed.csv'

    # set the column data types for memory efficiency
    coltypes_train = {'id': 'int64',
                      # 't_date':'datetime64',
                      'store_nbr': 'int8',
                      'item_nbr': 'int32',
                      'unit_sales': 'float32',
                      'promotion': 'bool'
                      }

    coltypes_test = {'id': 'int64',
                     # 't_date':'datetime64',
                     'store_nbr': 'int8',
                     'item_nbr': 'int32',
                     # 'unit_sales':'float64'
                     'promotion': 'bool'
                     }

    start_time = time.time()
    df_train = pd.read_csv(directory + file_train, dtype=coltypes_train, parse_dates=['date'],
                           infer_datetime_format=True)
    print_elapsed_time(printStoreNum + 'Finish Train file load', time.time() - start_time)

    start_time = time.time()
    df_test = pd.read_csv(directory + file_test, dtype=coltypes_test, parse_dates=['date'], infer_datetime_format=True)
    print_elapsed_time(printStoreNum + 'Finish Test file load', time.time() - start_time)


    #================
    # DATA MASSAGING
    # ================
    print(printStoreNum + 'Start Data Massaging')

    # Set the index to the date time
    df_train.set_index('date', inplace=True)
    df_test.set_index('date', inplace=True)

    # Filter the data.  We are only pulling back from August of last year for now to avoid the earthquake skew in April 2016
    # 12/11/2017 - Ben Grauer - commenting out with new pre-processed / shortened file created to load faster
    #df_train = df_train["2016-08-01":]

    # now Order the data for being clean
    df_train.sort_values(['store_nbr', 'item_nbr'], ascending=[True, True], inplace=True)

    # Re-initialize variables for function below
    # 12/11/2017 - Ben Grauer - commenting out with new pre-processed file
    if CONST_USE_SMALLER_MEMORY_FILE==True:
        idx = pd.DataFrame(pd.date_range('2016-08-01', '2017-09-01'), columns={'dateRange'})
        idx.set_index('dateRange', inplace=True)

    print(printStoreNum + 'Finished Data Massaging')

    #==========================
    # SET UP PARAM GRID SEARCH
    # =========================
    # Let's move to a grid search
    print(printStoreNum + 'Setting up parameter grid search')
    p = d = q = range(0, 2)
    param_pdq = list(itertools.product(p, d, q))

    # general all different combos of seasonal p, d, and q triplets
    param_seasonal_pdq3 = [(x[0], x[1], x[2], 3) for x in list(itertools.product(p, d, q))]
    param_seasonal_pdq7 = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
    param_seasonal_pdq14 = [(x[0], x[1], x[2], 14) for x in list(itertools.product(p, d, q))]
    param_seasonal_pdq31 = [(x[0], x[1], x[2], 31) for x in list(itertools.product(p, d, q))]

    # Stack each of the - took off the 31 to reduce # parameters / speed up scripts.
    # I want to put a yearly seasonality in there, but may need to put as a future item if time permits -
    #  would need to be on fruits/vegetables (if I can isolate)
    param_seasonal_pdq = param_seasonal_pdq3 + param_seasonal_pdq7 + param_seasonal_pdq14

    # If I want to include more parameters (had to remove some to make the evaluation complete faster)
    # By default leave at 0, meaning do not include/process
    runDeepEvaluation = 0
    if runDeepEvaluation == 1:

        # Include 31 days as an option
        param_seasonal_pdq + param_seasonal_pdq + param_seasonal_pdq31

        param_pdq = [item for item in param_pdq if item != (0, 0, 0) and item != (0, 1, 0)]

        # After analysis of preliminary runs:  Take these our from errors, and to cut back
        # Take out 000 - for fear of over-fitting or error out in the test data
        # Take out 0n0 - combinations from error
        #type(param_seasonal_pdq)
        param_seasonal_pdq = [item for item in param_seasonal_pdq if
                              item != (0, 0, 0, 3) and item != (0, 1, 0, 3) and
                              item != (0, 0, 0, 7) and item != (0, 1, 0, 7) and
                              item != (0, 0, 0, 14) and item != (0, 1, 0, 14)
                              ]

    print(printStoreNum + 'Finished parameter grid setup')

    #============================
    # ASSIGN THE MAIN LOOP ARRAY
    # ============================
    # NEED TO GRAB INPUTS FROM FILE HERE
    # MASSIVE LOOPING - Need to individualize these down to the store ID
    # Grab the arrays of uniqueness factors
    print(printStoreNum + 'Determining Starting Point')
    #TEST_REC_TO_LOOP = 10
    TEST_REC_TO_LOOP = file_args_num_test_rec


    df_test_iteration = pd.DataFrame(np.unique(df_test[['store_nbr','item_nbr']], axis=0), columns=('store_nbr','item_nbr'))
    df_test_iteration = df_test_iteration[df_test_iteration['store_nbr']==file_args_store_nbr]
    # df_test_iteration.sort_values(['store_nbr', 'item_nbr'], ascending=[True, True], inplace=True)
    # moved below

    #===========================
    # Here we need to determine where we left off in case the process gets cut, and then filter the data-set accordingly
    # ===========================

    # Call the function to determine where we left off
    if fn_determine_file_exists(exportResultsSubmissionFileName) == True:

        # Grab the completed results
        completed_df = fn_determine_file_last_run(exportResultsSubmissionFileName)

        # Should join back to test.  If there are no records to predict, then we will scoop in the end and leave at 0.
        # (meaning skip over).

        # Join the test_iteration
        df_resume = pd.merge(df_test_iteration, completed_df, how='left', on=('store_nbr', 'item_nbr'))
        df_resume['processed'].replace({np.nan: 0}, inplace=True)
        # Not really needed - sort used for testing /debugging this function / resume
        df_resume.sort_values(['store_nbr', 'item_nbr'], ascending=[True, True], inplace=True)

        # Then filter down the data-set to only un-filtered
        df_test_iteration = df_resume[df_resume['processed']==0]

        # set to True for below
        resumeRunningPreviousFile = True

        print(printStoreNum + 'Existing File Detected')

    # Re-Order
    df_test_iteration.sort_values(['store_nbr', 'item_nbr'], ascending=[True, True], inplace=True)

    # if we are testing a sub-set
    if TEST_REC_TO_LOOP > 0:
        df_test_iteration = df_test_iteration.head(TEST_REC_TO_LOOP)

    # reset the index
    df_test_iteration.reset_index(drop=True, inplace=True)

    # Set the full array to loop through
    fullArr = df_test_iteration.values

    #===========================
    # RUNNING OF THE PARAM TEST
    # ===========================
    print(printStoreNum +'Running TS Parameter Options')

    # Log
    logger.info('Total Items to Process: ' + str(len(fullArr)))


    #NUM_OF_WORKERS = 6 # Set the threads to 2 for this script (as we will run it under multiple single processes
    # This ended up being abandoned for the multi-process architecture
    NUM_OF_WORKERS = file_args_num_threads
    start_time = time.time()
    # run array with threads - changed this to single thread - for file appending - will be split by store_nbr to split up between systems
    #df_iterator = fn_multithread_TS_param_config(fn_loop_timeseries_param_options, fullArr, NUM_OF_WORKERS)

    # Flags to determine if we include the header in the
    includeSubmissionHeaderRunOnce = True
    includeParamHeaderRunOnce = True

    # If we are resuming a file, we don't need to include a header
    if resumeRunningPreviousFile == True:
        includeSubmissionHeaderRunOnce = False
        includeParamHeaderRunOnce = False


    print(printStoreNum +'Total items to process: ' + str(len(fullArr)))
    for i in range(0, len(fullArr)):

        # Set a print / log message
        if i % 200==0:
        # if i % 5==0:
            print(printStoreNum +'Current store loop index is: ' + str(i) + ' out of ' + str(len(fullArr)))


        # if no data exists in the train set, then skip looking for parameters or anything to speed up resuming a store
        if len(df_train[(df_train['store_nbr']==fullArr[i][0]) & (df_train['item_nbr']==fullArr[i][1])]) > 0:

            #====================================================
            # Obtain the ts Parameters for the store / item combo
            # ===================================================
            dfParameters = fn_loop_timeseries_param_options(fullArr[i], 0)
            # ====================================================

            # Only deal with parameters where data is present (filtering out values of "NO" and "ERROR")
            dfParameters = dfParameters[dfParameters['data_present']=='YES']

            if len(dfParameters) > 0:

                # Sort by the AIC
                dfParameters.sort_values(['model_aic'], ascending=[True], inplace=True)
                dfMinAICParameter = dfParameters.head(1) # Take min value

                # Append the order + seasonal Order to the arguments
                #CONST_TS_DAY_START = 300
                #minAICArr = fn_concat_args(fullArr[i][0], fullArr[i][1], dfMinAICParameter['model_order'], dfMinAICParameter['model_seasonal_order'], CONST_TS_DAY_START)
                dfMinAICParameter['ts_slice_start'] = 300
                minAICArr = dfMinAICParameter[['store_nbr', 'item_nbr', 'model_order', 'model_seasonal_order', 'ts_slice_start']].values

                # ====================================================
                # Run the time series for the minimal item from the search prior
                # ===================================================
                # df_EstimatesRun = fn_config_timeseries(fullArr[i])
                df_EstimatesRun = fn_config_timeseries(list(minAICArr[0]))

                # Grab dates 2017-08-15 - 2017-08-31 and write to the submission results file.
                df_EstimatesRun.reset_index(inplace=True)
                # rename column
                df_EstimatesRun = df_EstimatesRun.rename(columns={'index': 'date'})
                df_EstimatesRun = df_EstimatesRun[['id', 'date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion', 'forecast', 'forecast_rnd']]

                # Round for issues / Data Format
                df_EstimatesRun['id'] = df_EstimatesRun['id'].astype(int)
                df_EstimatesRun['store_nbr'] = df_EstimatesRun['store_nbr'].astype(int)
                df_EstimatesRun['item_nbr'] = df_EstimatesRun['item_nbr'].astype(int)
                df_EstimatesRun['unit_sales'] = df_EstimatesRun['unit_sales'].round(4)

                # Write the submissions results
                with open(exportResultsSubmissionFileName, 'a') as f:
                    df_EstimatesRun.to_csv(f, header=includeSubmissionHeaderRunOnce, index=False, quotechar='"')
                    f.close()
                    includeSubmissionHeaderRunOnce = False

            # Write to both files
            # Append the file for the parameter results, in case I want to pull later for lowest MSSE instead of lowest AIC
            with open(exportParamOptionsFileName, 'a') as f:
                dfParameters.to_csv(f, header=includeParamHeaderRunOnce, index=False, quotechar='"')
                f.close()

                # Toggle to False after the two runs
                includeParamHeaderRunOnce = False

        else:
            # log a skip
            logger.info('Skiped Item: ' + str(file_args_store_nbr) + ' - ' + str(fullArr[i][1]) + ' - ' + print_elapsed_time('Time Finished: ', time.time() - start_time, 1))
            # example
            #logger.info('Stop Processing Item: ' + str(arg_itemNumber) + ' - ' + print_elapsed_time('Time Finished: ', time.time() - start_time, 1))

    # Log
    logger.info('Total Store/Items Processing Time: ' + str(file_args_store_nbr) + ' - ' + print_elapsed_time('', time.time() - start_time))

    # We are DONE!
    print_elapsed_time(printStoreNum +'Finished store looping', time.time() - start_time)