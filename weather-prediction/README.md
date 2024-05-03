# Weather predicition Google Colab links

The weather prediction LSTM models were trained to predict SWC_5 given the feature engineered dataset for stations 1-6 using the feature engineering with rain chance calculator python notebook. The weather prediction folder has a folder for the feature engineered stations containing the datasets with the added columns with the past sums of 1 hour, 2 hours, 5 hours, 10 hours, and 24 hours, and then the predicted percentages for 1-7 days in the future for each hour of data. There were 2 ways created to approach creating the predicted percentages: creating a method to calculate hte numbers based on different sums of past hours and then ranges over time and then scaling for future days vs creating a random first model to incorporate seasonality to analyze the data and then create new columns into the dataframe with the predicted future days. 

Data Exploration Notebook: https://colab.research.google.com/drive/1wV_b5Z6-epJlju9Mt1kVSxfI63AIB1uO#scrollTo=guaWDILz1vzi

Predictive Models Gradient desent + exponential smoothing: https://colab.research.google.com/drive/1MT7dlUN39mTlJyoRB7HzRrWKXkzL3dpT?usp=sharing

Feature Engineering with calculate rain chance algorithm: https://colab.research.google.com/drive/1tAwbFKejNGi47uROgUNG_NRBwd6A-dCD?usp=sharing


bi_lstm_feedback models edited: https://colab.research.google.com/drive/1QsIfkHASqDCAhaFsSjYeKgb1B-dfq-L2?usp=sharing

Visualizations: https://colab.research.google.com/drive/1NjupBT-J-wcbXVFomxIXejWPlxvh7-lt#scrollTo=HkGd4dqxd2oS
Data:

Error Values and Poster Visualizations Doc: https://docs.google.com/document/d/1tb1mL3MsgzN2fsao-qj3bz9TcZXsXClSOUPzTzfICAo/edit?usp=sharing

Feature Engineering - Random Forest Model Dataset Generation: https://colab.research.google.com/drive/1ig5EQ6a4n0Q484PPS36KLQ-0TqEgqBp3?usp=sharing

Final Report: https://docs.google.com/document/d/1MP1peFpSTjY9-NHDyZBB0KoBdePH0npyRezLgGQXEUg/edit?usp=sharing
