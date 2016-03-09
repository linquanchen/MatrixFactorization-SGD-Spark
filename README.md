# MatrixFactorization-SGD-Spark

Spark is an efficient, general-purpose programming language to be used interactively to process large datasets on a cluster. This project focuses on using Apache Spark Programming Framework to optimize the sequential matrix factorization using Stochastic Gradient Descent(SGD) algorithm. Based on the sequential program(**oos-sgd-mf.py**), I implemented a basic program using spark and analyzed the disadvantages of this version. Then, I optimize the program using parallel updating the blocks, partition and group the blocks, and also tuning the spark configuration parameters to improve the performance.

