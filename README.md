# Recommender Systems

This project implements various types of recommender system techniques using Python3.

### Description

This project implements 3 primary types of recommender system techniques: Collaborative filtering, SVD and CUR matrix decompositions. Each one additionally also implements
the global baseline technique. The systems are evaluated using RMSE, precision on top K, and Spearman ranking.

### Prerequisites

Python3 along with nltk library is required to run this program.

### Structure of project

data_processing.py : Access the movie and ratings datasets and process them for usage in the various recommender systems.

collaborative_filtering.py : Implements the collaborative filtering recommender model. It is divided into 3 parts- precision on top K, Spearman ranking and RMSE
1. Precision on top K uses Pearson correlation for checking similarity between the test user and the other users and accordingly predicts ratings. It then calculates the precision among the top ratings.
2. Spearman ranking uses the Spearman ranking technique to predict the rating of a test user for some test movie.
3. RMSE uses Pearson correlation to calculate the similarities and then calculates the RMSE over some test dataset, which consists of randomly selected users accounting for
20% of the entire user database. It then calculates the RMSE from all the predicted ratings and displays a final RMSE value.

SVD.py : Recommends movies to users using SVD decomposition, it has the following methods within it,
1. SVD: Calculates the SVD decompostion of the matrix given to it.
2. Query: This function queries the SVD matrix given a query vector
3. RMSE: It calculates the root mean square error (RMSE) according to the definition given in the textbook for UV decomposition.
4. Energy: Calculates the energy of a matrix as defined as the sum of the squares of the individual elements
5. Precision_top_k: This calculates the Precision of the top K elements given a query and the V matrix of the SVD
6. spearmanCoefficient: Uses the Spearman ranking technique to predict the rating of a test user for some test movie.

CUR.py : Recommends users to movies using CUR decomposition, it has the following methods inside it,
1. CUR: It decomposes the given matrix into the three C,U,R matrices.
2. rmse: It calculates the root mean square error (RMSE) according to the definition given in the textbook for UV decomposition.
3. query: It takes a vector of user's previous ratings and predicts what movies to recommend to him/her.
4. precisionTopK: Precision on top K uses Pearson correlation for checking similarity between the test user and the other users and accordingly predicts ratings. It then calculates the precision among the top ratings.
5. Spearman ranking: Uses the Spearman ranking technique to predict the rating of a test user for some test movie.
6. Energy: this is a utility function used to calculate the energy of the decomposition.

### Run this in your project folder:

    $ python data_processing.py </br>
    $ python collaborative_filtering.py </br>
    $ python SVD.py </br>
    $ python CUR.py

### Results during runs

System | RMSE | Precision on top K | Spearman Ranking | Time
--- | --- | --- | --- | ---
Collaborative filtering | 0.9 | 72% | 40% | 1.823s
Collaborative filtering (baseline) | 3.6 | 65% | 26% | 1.823s
SVD | 0.18 | 78% | 85% |11.49ms
SVD (90% Energy) | 4.00 | 81% | 87% | 0.069ms
CUR | 0.112925 | 89.9% | 100% | 58.926 ms  
CUR (90% Energy) | 0.11166 | 89.9% | 100% | 51.930 ms

### Built With

The project uses

- Python3
- Numpy
- Scipy
- OS
- Pickle
- Math

### Authors

Chandrahas Aroori [https://github.com/Exorust] </br>
Naren Surampudi [https://github.com/nsurampu] </br>
Aditya Srikanth [https://github.com/aditya-srikanth]

### Acknowledgments

We'd like to thank our Information Retrieval instructor to give us this opportunity to make such a project.
