# Disaster-Response-Pipeline

## Background
Following a disaster, disaster response teams may recieve thousands and thousands of messages. It would be almost impossible for the team to carefully read every message. Can we use data science and machine learning to pull out the messages that are the most important?

## Goal
The goal of this project is to use prelabeled disaster messages to train a supervised machine learning algorithm to classify messages. This way, important messages can be shown to the correct team. 

## File Descriptions

### ETL 
data/disaster_categories.csv -- a unique identifier, the original message, the cleaned message, and the genre
data/disaster_messages.csv -- a unique identifier, and the categories related to that message
data/process_data.py -- an Extract, Transform, Load pipeline which writes cleaned data to data/DisasterResponse.db

### ML 
models/train_classifier.py -- load data from db into dataframe, find optimal parameters, train random forrest model, save model to .pkl file
models/classifier.pkl -- to generate this file, clone this repository, and run the following from the root of the project directory:

run the ETL pipeline that cleans data and stores in database:
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

run the ML pipeline that trains classifier and saves:
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
