# Taxi-Fare-Competition

Project submission for Foundations of Data Analytics (CS910) course at the University of Warwick, Decemebr 2018. The task was to become a 'Data analyst' and analyse and report findings on a chosen dataset. The dataset chosen was the Taxi Fare Competition dataset hosted on Kaggle [Taxi Fare Competition](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/overview).   

## Abstract of report 
The data set for the NYC Taxi Fare competition on Kaggle was explored and modelled in
order to build an algorithm which best predicts the values of taxi fares in NYC. Features
were extracted from the initial attributes of pickup, drop-off locations and timestamp for
the journey. As expected the most useful feature for predicting was found to be the
haversine distance between locations. The most successful model built was an extreme
gradient boosting classifier which scored an RMSE of 3.051 on the test data. The code
developed for the challenge can be found here.

## Structure of Repository

- `final_report.pdf` - Final report submitted for the course presenring the findings for the report.

- `/data` - Contains a link to the download page for the traning and test data provided by Kaggle for the competition.

- `/scripts` - Contains the scripts for the data processing and inference tasks.

- `/images` - Contains the images present in the final report.

- `/results` - Contains the output path for the results ran in for the classifier (all results were removed hence empty). 
