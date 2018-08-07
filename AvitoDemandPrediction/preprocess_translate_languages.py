# coding: utf-8
# path=C:\ProgramData\Anaconda3\

#=====================================================================================================
# Author: Ben Grauer
# Purpose: This script will take a file and translate it from Russian to English adding each
#  translated column as a new column (retaining the original).  Can resume on previous file
#
# Example Script Call - Running the short columns/grouped first / then long columns:
# python process_translate_languages.py "D:/project/data/kg_avito_demand/test.csv", 0
# python process_translate_languages.py "D:/project/data/kg_avito_demand/test.csv", 1
#=====================================================================================================

import pandas as pd
import numpy as np

from datetime import datetime
import math
import sys

import time

# look this up - this 
import mtranslate as trans
import textblob # try this as well - seems to be faster

from unidecode import unidecode
import codecs


# Function to Add Translation Columns to the dataframe (to keep original) or filter an already run file
def fn_AddTranslationColumnsORFilterExisting(df, columnsToConvertShortList, columnsToConvertLongList):
    
    # Create the translated dataframe column
    for i in columnsToConvertShortList:
        # check for existing
        if i + '_translated' in df:
            print('short list exists')
        else:
            df[i + '_translated'] = ''

    for i in columnsToConvertLongList:
        # check for existing
        if i + '_translated' in df:
            print('long list exists')
        else:
            df[i + '_translated'] = ''
            
    if 'processed' in df:
        print('processed exists') #, grabbing un-processed records only')
    
    else:
        df['processed'] = 0

    return df


# Function to be able to switch between conversions
def fn_TranslateWord(word):
    # with textblob
    try:
        return str(textblob.TextBlob(word).translate(from_lang="ru",to="en"))
    except:
        return word
    
    # with mttrans
    #return trans.translate(word, from_language='ru', to_language='en')


# You can flip this flag to over-write an existing column, or create a new one (depending on testing / size / what you want to do)
def fn_determineTransColumn(existingColumnName):
    
    # You can choose to flip this flag to override the existing column
    overWriteExistingColumn = 0
    
    if overWriteExistingColumn == 0:
        existingColumnName = existingColumnName + '_translated'
    elif overWriteExistingColumn == 1: 
        existingColumnName = existingColumnName


# This function will cycle through each grouped list (update columns in a grouped style, vs row by row - speed things up)
def fn_TranslateDataFrame_Grouped(df, columnsToConvert, exportFileName):

    print('Total Grouped colums to convert: ' + str(len(columnsToConvert)))

    for i in columnsToConvert:

        # Get unique values of the column we are able to translate
        arrUniqueValues = df[i].unique()

        print('\nProcessing column: ' + i + ' - Total Count: ' + str(len(arrUniqueValues)))
        print('   Translating Column Values: ' + i)

        # Set a dictionary of the unique values to be translated
        curDict = {}
        for j in arrUniqueValues:
            curDict[j] = fn_TranslateWord(j)

        # now we want to map the dictionary back to the dataframe 
        print('   Updating Column Values: ' + i)

        # update the translated column
        for key, value in curDict.items():
            df.loc[df[i] == key, (i + '_translated')] = value
        
        print('   Completed Updating Column Values: ' + i + '\n========================')

    # Writing Final File
    file = codecs.open(exportFileName, 'w', 'utf-8') 
    df.to_csv(file, index=False)
    file.close()
    
    print('Finished')

    return df


# Now do the same for the long list - except do this one by one / row by row.  Will hit each of the columns
def fn_TranslateDataFrame_Individual(df, columnsToConvert, exportFileName):

    print('Total Individual colums to convert: ' + str(len(columnsToConvert)))
    print('Total Individual row to convert: ' + str(len(df)))
    startTime = datetime.now()
    
    # Set the 
    max_count = len(df)
    
    # For each row in the data set
    for i, row in df.iterrows():

        # Here skip if we have already processed (or only process if still outstanding)
        if df.at[i, 'processed'] == 0:
        
            # now cycle through each of the individual columns at a time
            for colName in columnsToConvert:

                if pd.isnull(df.iloc[i][colName]):
                    df.iloc[i, df.columns.get_loc(colName + '_translated')] = pd.np.nan
                else:
                    df.iloc[i, df.columns.get_loc(colName + '_translated')] = fn_TranslateWord(df.iloc[i][colName])

                # Set the processed
                df.at[i, 'processed'] = 1

                # only run on the second column, not each column for the checkpoint
                if (i % 1000 == 0) and colName == 'description':
                    print('Processed row (' + str(datetime.now()) + '): ' + str(i) + ' and creating file snapshot. ' + str(datetime.now() - startTime))

                    file = codecs.open(exportFileName, 'w', 'utf-8') 
                    df.to_csv(file, index=False)
                    file.close()

        # Test 25
        #if i % 10 == 0:
        #    break

    # Writing Final File
    file = codecs.open(exportFileName, 'w', 'utf-8') 
    df.to_csv(file, index=False)
    file.close()
    
    print('Finished')

    # How Long
    print (datetime.now() - startTime)
        
    return df

def run_main_function(fileName, runLongOnly):

    df = pd.read_csv(fileName)

    # columns to convert
    shortColumnsToConvert = ['region','city','parent_category_name','category_name','param_1','param_2','param_3']
    longColumnsToConvert = ['title','description']
    
    # Add the new columns - if they do not exist yet
    df = fn_AddTranslationColumnsORFilterExisting(df, shortColumnsToConvert, longColumnsToConvert)
    
    # opt out of re-running short again (long only)
    if runLongOnly == 0:
	    df = fn_TranslateDataFrame_Grouped(df, shortColumnsToConvert, fileName)
		  
    df = fn_TranslateDataFrame_Individual(df, longColumnsToConvert, fileName)



# Main
if __name__ == "__main__": 

    if len(sys.argv) >= 2:
        fileNameToTranslate = sys.argv[1]
        print('Translating File: ' + fileNameToTranslate)
	
    if len(sys.argv) >= 3:
        longColumnsOnly = int(sys.argv[2])
        print('Long Columns Only ' + str(longColumnsOnly))
	
	# run the function
    run_main_function(fileNameToTranslate, longColumnsOnly)