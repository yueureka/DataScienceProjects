# Disaster Response Pipeline
Following a disaster, there are millions of communications from either direct or via social media. Different types of disaster response organizations take care of different parts of the disasters and observe messages to understand the needs of the situation. They have the least capacity to filter out messages during a large disaster, therefore to categorize each of these communications is critical for them to respond efficiently and effectively.
In this project, the dataset of communications with different categorical labels is given, ETL pipeline was built to clean the messages with regex and NLTK, then the pre-processed data was trained with machine learning pipeline, which is composed of tokenization, tfidf transformation, and multi-output classifier. Random forest classifier was applied as the core estimator, grid search was conducted for parameter optimization, prediction results were given via classfication_report function. See more details in the Jupyternotebook.   

# Installation
The code runs in Python 3, Python libraries like pandas, numpy and matplotlib are used in the code. 

# File Descriptions
process_data.py - ETL script to clean data into the proper format by splitting up categories and making new columns for each as target variables.

train_classifier.py - Script to tokenize messages from clean data and create new columns through feature engineering. The data with new features are trained with an ML pipeline and pickled.

run.py - Main file to run Flask app that classifies messages based on the model and shows data visualizations.

ETL Pipeline Preparation.ipynb - The notebook that shows steps of the ETL pipeline

Global Suicide Rate.ipynb - The notebook file that showcases the machine learning pipeline and reports

Messages.csv - data contains messages with categories.  

# Output
After running the app, the following screenshot will be shown:


# Licensing and Acknowledgements
The dataset is provided by Figure8, all rights reserved.  
