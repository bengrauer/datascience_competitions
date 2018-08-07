# Competition: Avito Demand Prediction Challenge
- Sub-Directory: /AvitoDemandPrediction/
- WebLink: https://www.kaggle.com/c/avito-demand-prediction
- Description: Predict demand for an online classified ad
- Final Place: 1,678 out of 1,917 June 2018
- Code/Model: Python | Random Forest Regressor
- Size: 2MM obs. Images - 70 GB

## Files:
- README.md - this file.  Description of directory contents
- preprocess_translate_languages.py - python script to translate an entire file to another language (English)
- preprocess_createlist_groupedImages.ipynb - Notebook to read all files in a directory and split into quartiles for processing in parallel
- preprocess_images.ipynb - Notebook to extract features from images
- preprocess_text.ipynb - Notebook to gather basic text aggregates from ad description 
- helper.py - Script to help load and merge all of the pre-processed data
- plot_data_1.ipynb - Notebook for plotting of the data
- models_generalmodels.ipynb - Notebook for general model processing

## Quick Description
Avito is an online platform for selling used goods, and the challenge was to predict the probability of demand for an online advertisement.  Like Craig's List, but in Russia.  My biggest take away from this competition was really all the coding / pre-processing of scripts to use in the future: 1) translate foreign language, 2) obtain features from images 3) general text/word aggregates and parsing, and 4) pragmatic approach to swapping out different algorithms.