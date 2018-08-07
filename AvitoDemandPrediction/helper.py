# coding: utf-8
# path=C:\ProgramData\Anaconda3\

#=====================================================================================================
# Author: Ben Grauer
# Purpose: General script to load data (to reduce code across plotting + model notebooks)
#
#=====================================================================================================


import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from datetime import datetime
import time

import math
import mtranslate as trans

#from ipywidgets import FloatProgress
#from IPython.display import display


def main():
	file=sys.argv[1]
	dest=sys.argv[2]

	
# Function to add aggregate stats to the data frame
def fn_AddStatToDataframe(df, groupColumnList, columnName, statistic):

    if statistic == 'mean':
        df = pd.DataFrame(df.groupby(groupColumnList)[columnName].mean())
    elif statistic == 'std':
        df = pd.DataFrame(df.groupby(groupColumnList)[columnName].std())
    elif statistic == 'var':
        df = pd.DataFrame(df.groupby(groupColumnList)[columnName].variance())
        
    df.rename(columns={columnName:columnName + '_' + statistic}, inplace=True)
    df.round(decimals=4)
    df.reset_index(inplace=True)
    df.head()
    
    return df

# Merges together all the pre-processed files
def fn_AppendAllData(dfInputTrain, dfInputTest, startTime):

	# Rest of code
	workDir = 'D:/project/data/kg_avito_demand/'

	print('Loading Train Image Stat Files (3-splits)')
	dfImage1 = pd.read_csv(workDir + 'FullImageFileList_1.csv')
	dfImage2 = pd.read_csv(workDir + 'FullImageFileList_2.csv', header=0)
	dfImage3 = pd.read_csv(workDir + 'FullImageFileList_3.csv', header=0)
	frames = [dfImage1, dfImage2, dfImage3]
	dfImageTrain = pd.concat(frames)
	dfImageTrain['image'] = dfImageTrain.apply(lambda row: str(row['imageFileName']).replace('d:/project/data/kg_avito_demand/train_jpg\\','').replace('.jpg',''), axis=1)

	print (datetime.now() - startTime)
	print('Loading Test Image Stat Files')
	dfImage4 = pd.read_csv(workDir + 'FullImageFileList_4.csv')
	dfImageTest = dfImage4
	dfImageTest['image'] = dfImageTest.apply(lambda row: str(row['imageFileName']).replace('d:/project/data/kg_avito_demand/test_jpg\\','').replace('.jpg',''), axis=1)

	print (datetime.now() - startTime)
	print('Loading Text Stat Files')
	dfTextTrain = pd.read_csv(workDir + 'FullTextProcessedTrain_1.csv')
	dfTextTest = pd.read_csv(workDir + 'FullTextProcessedTest_1.csv')

	print (datetime.now() - startTime)
	print('Merging Image Stat Files')
	dfInputTrain = pd.merge(dfInputTrain, dfImageTrain, how='left', on='image')
	dfInputTest = pd.merge(dfInputTest, dfImageTest, how='left', on='image')

	print (datetime.now() - startTime)
	print('Mergeing Text Stat Files')
	textColumns = [\
	'item_id','desc_numWords','desc_numStopWords','desc_numNouns',\
	'desc_numVerbs','desc_numAdjs','desc_numSymbols','desc_wordsCondition',\
	'desc_wordBargainOrDeal','desc_numNonASCIIWords','desc_numNumericWords','desc_numUpperCaseWords',\
	'desc_avgImportantWordLength','desc_avgAllWordLength','desc_numSentences','desc_avgWordsPerSentence',\
	'desc_avgWordLengthPerSentence']
	dfInputTrain = pd.merge(dfInputTrain, dfTextTrain[textColumns], how='left', on='item_id')
	dfInputTest = pd.merge(dfInputTest, dfTextTest[textColumns], how='left', on='item_id')
	print('Finished Loading and Merging Image and Text Files')
	print (datetime.now() - startTime)
	
	return dfInputTrain, dfInputTest
	
def fn_LoadData(trainFileName, testFileName):
	
	#workDir = 'D:/project/data/kg_avito_demand/'
	#fileTest = 'adj_test_en.csv'
	#fileTrain = 'adj_train_en.csv'	
	
	startTime = datetime.now()
	
	print('Loading Train and Test Files')
	dfTrain = pd.read_csv(trainFileName)
	dfTest = pd.read_csv(testFileName)
	
	# rename the files.  The original (foreign languge items) are renamed to "_orig"
	dfTrain.rename(columns={\
	'region':'region_orig', \
	'city':'city_orig', \
	'parent_category_name':'parent_category_name_orig', \
	'category_name':'category_name_orig', \
	'param_1':'param_1_orig', \
	'param_2':'param_2_orig', \
	'param_3':'param_3_orig', \
	'title':'title_orig', \
	'description':'description_orig', \
	}, inplace=True)

	# And then rename the translated columns to the standard names (to keep model columns named in a shorter fashion
	dfTrain.rename(columns={\
	'region_translated':'region', \
	'city_translated':'city', \
	'parent_category_name_translated':'parent_category_name', \
	'category_name_translated':'category_name', \
	'param_1_translated':'param_1', \
	'param_2_translated':'param_2', \
	'param_3_translated':'param_3', \
	'title_translated':'title', \
	'description_translated':'description' \
	}, inplace=True)
	
	dfTest.rename(columns={\
	'region':'region_orig','region_translated':'region', \
	'city':'city_orig', 'city_translated':'city', \
	'parent_category_name':'parent_category_name_orig', 'parent_category_name_translated':'parent_category_name', \
	'category_name':'category_name_orig', 'category_name_translated':'category_name', \
	'param_1':'param_1_orig', 'param_1_translated':'param_1', \
	'param_2':'param_2_orig', 'param_2_translated':'param_2', \
	'param_3':'param_3_orig', 'param_3_translated':'param_3', \
	'title':'title_orig', 'title_translated':'title', \
	'description':'description_orig', 'description_translated':'description' \
	}, inplace=True)

	# re-order / possible drop (or keep in)
	dfTrain = dfTrain[['item_id','user_id','region','city','parent_category_name','category_name','param_1','param_2','param_3','title','description','price','item_seq_number','activation_date','user_type','image','image_top_1','deal_probability','region_orig','city_orig','title_orig','description_orig']]
	dfTest = dfTest[['item_id','user_id','region','city','parent_category_name','category_name','param_1','param_2','param_3','title','description','price','item_seq_number','activation_date','user_type','image','image_top_1','region_orig','city_orig','title_orig','description_orig']]
	
	groupColumnList = ['region','city','category_name','user_type']
	joinColumns = ['region','city','category_name','user_type']

	print (datetime.now() - startTime)
	print('Adding Stats Features')
	
	# Grab counts to verify below
	totalTrainRecords = len(dfTrain.index)
	totalTestRecords = len(dfTest.index)
	print('Total Train Records: ' + str(totalTrainRecords))
	print('Total Test Records: ' + str(totalTestRecords))

	# ADD Statistics for the main groupings above
	# Price Mean / # Price std
	dfTrain = pd.merge(dfTrain, fn_AddStatToDataframe(dfTrain, groupColumnList, 'price', 'mean'), how='left', on =joinColumns)
	dfTrain = pd.merge(dfTrain, fn_AddStatToDataframe(dfTrain, groupColumnList, 'price', 'std'), how='left', on =joinColumns)
	
	dfTest = pd.merge(dfTest, fn_AddStatToDataframe(dfTest, groupColumnList, 'price', 'mean'), how='left', on =joinColumns)
	dfTest = pd.merge(dfTest, fn_AddStatToDataframe(dfTest, groupColumnList, 'price', 'std'), how='left', on =joinColumns)
	
	# ImageTop Mean / Std
	dfTrain = pd.merge(dfTrain, fn_AddStatToDataframe(dfTrain, groupColumnList, 'image_top_1', 'mean'), how='left', on =joinColumns)
	dfTrain = pd.merge(dfTrain, fn_AddStatToDataframe(dfTrain, groupColumnList, 'image_top_1', 'std'), how='left', on =joinColumns)
	dfTest = pd.merge(dfTest, fn_AddStatToDataframe(dfTest, groupColumnList, 'image_top_1', 'mean'), how='left', on =joinColumns)
	dfTest = pd.merge(dfTest, fn_AddStatToDataframe(dfTest, groupColumnList, 'image_top_1', 'std'), how='left', on =joinColumns)
	
	# DealProb Mean / Std
	dfTrain = pd.merge(dfTrain, fn_AddStatToDataframe(dfTrain, groupColumnList, 'deal_probability', 'mean'), how='left', on =joinColumns)
	dfTrain = pd.merge(dfTrain, fn_AddStatToDataframe(dfTrain, groupColumnList, 'deal_probability', 'std'), how='left', on =joinColumns)

	# SeqNum Mean
	dfTrain = pd.merge(dfTrain, fn_AddStatToDataframe(dfTrain, groupColumnList, 'item_seq_number', 'mean'), how='left', on =joinColumns)
	dfTest = pd.merge(dfTest, fn_AddStatToDataframe(dfTest, groupColumnList, 'item_seq_number', 'mean'), how='left', on =joinColumns)

	# Clean any data
	print (datetime.now() - startTime)
	print('Cleaning data')
	dfTrain[['title','title_orig','description','description_orig']] = dfTrain[['title','title_orig','description','description_orig']].fillna('')
	dfTest[['title','title_orig','description','description_orig']] = dfTest[['title','title_orig','description','description_orig']].fillna('')
	
	
	# FEATURES
	# Determine if there was any instance
	
	# Add Binary Outcome Classifier
	dfTrain.loc[dfTrain['deal_probability'] > 0, 'deal_binary'] = 1
	dfTrain.loc[dfTrain['deal_probability'] <= 0, 'deal_binary'] = 0
	
	# Now Load the Text + Image DataFrame
	print (datetime.now() - startTime)
	print ('Loading Image + Text Data')
	dfTrain, dfTest = fn_AppendAllData(dfTrain, dfTest, startTime)

	# Some more derivations
	# Blur Quality
	dfTrain['blur_quality'] = ''
	dfTrain.loc[dfTrain['blurColorScale'] <= 300, 'blur_quality'] = 'poor'
	dfTrain.loc[(dfTrain['blurColorScale'] > 300) & (dfTrain['blurColorScale'] <= 400), 'blur_quality'] = 'average'
	dfTrain.loc[dfTrain['blurColorScale'] > 400, 'blur_quality'] = 'good'
	dfTest['blur_quality'] = ''
	dfTest.loc[dfTest['blurColorScale'] <= 300, 'blur_quality'] = 'poor'
	dfTest.loc[(dfTest['blurColorScale'] > 300) & (dfTest['blurColorScale'] <= 400), 'blur_quality'] = 'average'
	dfTest.loc[dfTest['blurColorScale'] > 400, 'blur_quality'] = 'good'
	
	# TODO: Improve with historgram + centroid
	# Determine if image is mostly white (pre-fabricated picture with white background)
	dfTrain.loc[dfTrain['brightness'] >= 250, 'image_whitespace'] = 1
	dfTrain.loc[dfTrain['brightness'] < 250, 'image_whitespace'] = 0
	dfTest.loc[dfTest['brightness'] >= 250, 'image_whitespace'] = 1
	dfTest.loc[dfTest['brightness'] < 250, 'image_whitespace'] = 0
	
	# Image Present
	dfTrain['image_present'] = 0
	dfTrain.loc[pd.notna(dfTrain['image']), 'image_present'] = 1
	dfTest['image_present'] = 0
	dfTest.loc[pd.notna(dfTest['image']), 'image_present'] = 1
	
	# numerical check
	totalSuperRecords = len(dfTrain.index)
	if totalTrainRecords != totalSuperRecords:
		print('Records do not match!! ' + 'Train: ' + str(totalTrainRecords) + '  Super: ' + str(totalSuperRecords))
	else:
		print(':-) Train: ' + str(totalTrainRecords) + '  Super: ' + str(totalSuperRecords))
		
    # return the merged / completed data frames
	return dfTrain, dfTest

if __name__ == '__main__':
    main() 

