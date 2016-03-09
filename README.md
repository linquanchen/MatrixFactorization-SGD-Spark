# MatrixFactorization-SGD-Spark

Spark is an efficient, general-purpose programming language to be used interactively to process large datasets on a cluster. This project focuses on using Apache Spark Programming Framework to optimize the sequential matrix factorization using Stochastic Gradient Descent(SGD) algorithm. Based on the sequential program(**oos-sgd-mf.py**), I implemented a basic program using spark and analyzed the disadvantages of this version(***mf.py***). Then, I optimize the program using parallel updating the blocks, partition and group the blocks, and also tuning the spark configuration parameters to improve the performance.

## Data
* Download data: Read the usage license at [grouplens](http://grouplens.org/datasets/movielens) to better understand the limitations on use of this data.

```
1M: 
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
10M : 
wget http://files.grouplens.org/datasets/movielens/ml-10m.zip
22M : 
wget http://files.grouplens.org/datasets/movielens/ml-latest.zip
```
* Formate data

```
1M:
unzip ml-1m.zip ; cut -d':' -f1,3,5 ml-1m/ratings.dat | sed 's/:/,/g' > ratings_1M.csv
10M:
unzip ml-10m.zip ; cut -d':' -f1,3,5 ml-10M100K/ratings.dat | sed 's/:/,/g' > ratings_10M.csv
22M:
unzip ml-latest.zip ; cut -d',' -f1,2,3 ml-latest/ratings.csv > ratings_22M.csv
```

## Cluster
* create cluster: You can use the EMR of Amazon Web Service to create a Spark cluster with one master and two slaves, choose the instance type: **c4.xlarge**
*  upload file: upload the **my.py** and **xxx.csv** into master node
*  put data file to HDFS: such as ```hadoop fs -put xxx.csv```

## How to Run
* Run the program 
```
[time] spark-submit --master yarn [--spark-config-params] mf.py {dataset-location} {rank} {output-location-w} {output-location-h}
```
For example:
```
time spark-submit --executor-memory 10G mf.py ratings_1M.csv 30 w.csv h.csv
```

* Calculate the RMSE
```
python rmse.py {w-location} {h-location} {data-location}
```

 For example:
```
python rmse.py w.csv h.csv ratings_1M.csv
```

 **PS: You need to get the output file from HDFS**
 




