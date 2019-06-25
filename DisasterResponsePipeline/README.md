# Disaster Response Pipeline
Following a disaster, there are millions of communications from either direct or via social media. Different types of disaster response organizations take care of different parts of the disasters and observe messages to understand the needs of the situation. They have the least capacity to filter out messages during a large disaster, therefore to categorize each of these communications is critical for them to respond efficiently and effectively.

In this project, the dataset of communications with different categorical labels is given, ETL pipeline is built to clean the messages with regex and NLTK, then the pre-processed data is trained with machine learning pipeline, which is composed of tokenization, tfidf transformation, and multi-output classifier. Random forest classifier is applied as the core estimator, grid search is conducted for parameter optimization, prediction results are given via classfication_report function. See more details in the Jupyternotebook.   

# Installation
The code runs in Python 3, Python libraries like pandas, numpy and matplotlib are used in the code, nltk library is aslo installed and used for language processing.  

# File Descriptions
data/process_data.py - ETL script to clean data into the proper format by splitting up categories and making new columns for each as target variables.

models/train_classifier.py - Script to tokenize messages from clean data and create new columns through feature engineering. The data with new features are trained with an ML pipeline and pickled.

App/run.py - Main file to run Flask app that classifies messages based on the model and shows data visualizations.

ETL Pipeline Preparation.ipynb - The notebook that shows steps of the ETL pipeline

Global Suicide Rate.ipynb - The notebook file that showcases the machine learning pipeline and reports

Messages.csv - data contains messages with categories.  

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
![alt text](https://github.com/yueureka/DataScienceProjects/blob/master/DisasterResponsePipeline/disaster-response-project2.png)

# Licensing and Acknowledgements
The dataset is provided by Figure8, all rights reserved.  
