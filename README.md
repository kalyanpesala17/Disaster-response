# Project Motivation
In the times of uncertainity like disaster we get a lot of messages for help from the various sources(news,social media, direct responses) to effectively distribute the resources, we analyze the disaster data from Figure Eight to build a model for an API that classifies disaster messages.it will include a web app where an emergency worker can input a new message and get classification results in several categories to distribute the resources where it needed the most. The web app will also display visualizations of the data.

## Table of Contents
1. [Installation](#installation)
2. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Results](#results)

### Installation <a name="installation"></a>
For running this project, the most important library is Python version of Anaconda Distribution. It installs all necessary packages for analysis and building models. 


### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/


### File Descriptions <a name="files"></a>
1. data/process_data.py: The ETL pipeline used to process and clean data in preparation for model building.
2. models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle.
3. app/templates/*.html: HTML templates required for the web app.
4. app/run.py: To start the Python server for the web app and render visualizations.

### Results<a name="results"></a>
The main observations of the trained classifier can be seen by running this application. below are the screenshots of webapp.
![Images of web app](https://github.com/kalyanpesala17/Disaster-response/blob/master/Screenshot%20(205).png)
![Images of web app](https://github.com/kalyanpesala17/Disaster-response/blob/master/Screenshot%20(206).png)

