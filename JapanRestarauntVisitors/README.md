## Recruit Restaurant Visitor Forecasting
- Sub-Directory: /JapanRestarauntVisitors/
- WebLink: https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting
- Final Place: 1,747 out of 2,158
- Code/Model: Python | ARIMA 
- Description: Predict the number of visitors 45 days out for several local restaraunts in Japan.

## Files:
- plot_QuickAnalysis.ipynb - Notebook of quick analysis and plotting
- helper_notebook.ipynb - Helper notebook to isolate functions, data loading, formatting, etc
- script_ARIMA.ipynb - Final notebook / script used for the best competition scores
- submission_log.txt - log of all submissions.  

## Description of Script
Ended up using a simple ARIMA model.  The helper notebook would perform the data loading, and the ARIMA script would run the models.  Experimented a little bit with a multi-variate model and multi-STEP in the earlier stages, ended with standard ARIMA as it was producing better scores.