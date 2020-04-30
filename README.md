# Unsupervised Final Project by Gal Eyal

##About this project
This repository contains all the code, figures and results of my final project at the course "Supervised and Unsupervised Learning" at Bar-Ilan University.\
The project solves the problem of predicting a movie genre using ratings data sets with different sparsity levels.\
The original data should be downloaded from Kaggle and stored in the data folder.\
Download link: https://www.kaggle.com/rounakbanik/the-movies-dataset\
The relevant files are: movies_metadata.csv, ratings.csv and links.csv

##Usage
The full run of the code includes data set preparation and different experiments which can take couple of hours.
The git already consist the pre computed data sets as pickles in the data folder as well as all experiments results in the results folder.
Appropriate flags can be used in order to avoid re-computation for example:

Run the code using all pre-computed files (fastest) :
>python main.py --skip-dataset-creation --skip-nmf-experiment --skip-mds-experiment

Run the full code (longest):
>python main.py --metadata-file ./data/movies_metadata.csv --ratings-file ./data/ratins.csv --links-file ./data/lins.csv

** notice that re-running data creation or experiments overwirghts the pre-computed files
(In case of an error\early stop of the run it is always possible to download the pre-computes results again).

For more options and help:
>python main.py -h

##Main packages dependencies:
- pandad
- sklearn
- numpy
- scipy
- matplot lib
- tqdm