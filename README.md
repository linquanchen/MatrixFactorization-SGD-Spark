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
 
## Algorithms
As shown in Figure 1, I updated the blocks in the diagonal order, which means that I updated the block 1(red), block 2(yellow), block 3(black) and block 4(green) at same time respectively. We can know from the figure 1, every color blocks correspond to different parts of H and W blocks, therefore we can update the H and W matrices parallel and not affect each other.

![dada](https://raw.githubusercontent.com/linquanchen/MatrixFactorization-SGD-Spark/master/update-blocks-in-diagonal.png)
**Figure 1. Updating blocks in diagonal**

For more details about the algorithms and the optimization, please see the [document!](Explore and Optimize Matrix Factorization with Spark_Linquan_Chen.pdf).

## Future Plan
The most challenge of my program is about memory. I think there are two steps to solve the problem: 

* Firstly, I need to optimize my program. Because, I have created many large RDDs during the iterative processes. I need to discard them when no longer need them. Therefore, I need to re-analyze the process and optimize the memory utilization. 
* Secondly, I can optimize the program at the aspect of tuning Spark Configuration parameters, such as cluster instance type and number, executor memory and number, driver memory and number, memory fraction and so on.





