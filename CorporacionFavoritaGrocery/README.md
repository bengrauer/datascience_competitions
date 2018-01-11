# Competition: Corporación Favorita Grocery Sales Forecasting
- Sub-Directory: CorporacionFavoritaGrocery
- WebLink: https://www.kaggle.com/c/favorita-grocery-sales-forecasting
- Current Place: 1,237 out of 1,613 (contest finalizes mid Jan/2018))
- Description: Predict 3,900 grocery item sales across 54 stores out 30 days.

## Files:
- plot_0_QuickAnalysis.ipynb - Quick Analysis / Plotting notebook
- script_1_CreateSubFile.ipynb - Notebook to trim the main training data (explanation enclosed in script)
- GenTimeSeriesOptionsAndResults.py - Main python script that generates the time-series options and results
- InvokeMultipleProcesses.py - Python script that was a multi-process wrapper / handler.  I would call this script, which started multiple instances of "GenTimeSeriesOptionsAndResults.py".  
- script_4_FinalSubmissionFileGeneration.ipynb - This notebook would gather and merge all the individual store submission files into the final submission file.
- script_5_FinalParameterReruns.ipynb - Notebook that would re-run any variant of the parameters from the original hyperparameter search. In case I wanted to make a slight adjustment and re-submit.

## Description of Script
This was a first attempt at a time-series problem.  It quickly turned into a big-data problem as I realized that the grocery item sales numbers were significantly different across stores.  In trying to compete with the best possible score I utilized a hyperparameter search across each grocery item/store combination.  The scripts are capable of running specific grocery stores across multiple computers in a manual/distributed fashion.  I was planning on utilizing an ARIMAX model because there were oil exports, but with all additional hurdles, the model only utilized the single variable of grocery sales units.

Learning Takeaways: 1) Utilization of multi-process python scripting to break up the load, 2) AWS server setup to off-load computational processing (including utilizing spot instances), 3) Auto-resuming processes.

There are a couple of comments that are dated with my name.  That is just my commenting format/style (date + name).  All scripts are original.