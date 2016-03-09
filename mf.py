#!/usr/bin/env python3

# Author: Linquan Chen
# E-mail: linquanc@andrew.cmu.edu
# Updated: 09/03/2016 17:00:00

# exmaple command to run:
# spark-submit --master yarn --executor-memory 10G mf.py ratings_1M.csv 30 w.csv h.csv

import sys
import numpy as np
import random
import time
from pyspark import SparkContext
import pyspark
#import logging

# Using SGD to update the W and H matrices
def sgd_on_one_block_func(step_size, x_block_dim, y_block_dim, R_rows, W_rows, H_rows):
    W_rows = list(W_rows)[0]
    H_rows = list(H_rows)[0]
    result = []
    # Iterate every block
    for x in R_rows:
        x_id = int(x[0])
        y_id = int(x[1])
        rating = float(x[2])
         
        diff = rating - np.dot(W_rows[x_id % x_block_dim], H_rows[y_id % y_block_dim])
        W_gradient = np.multiply(-2 * diff, H_rows[y_id % y_block_dim])
        W_rows[x_id % x_block_dim] -= step_size * W_gradient
     
        H_gradient = -2 * diff * W_rows[x_id % x_block_dim]
        H_rows[y_id % y_block_dim] -= step_size * H_gradient
    result.append(W_rows)
    result.append(H_rows)
    return result

# Evalute the RMSE of blocks in diagonal
def evaluate_on_one_block_func(x_block_dim, y_block_dim, R_rows, W_rows, H_rows):
    W_rows = list(W_rows)[0]
    H_rows = list(H_rows)[0]
    err = 0.0
    n = 0
    for x in R_rows:
        x_id = int(x[0])
        y_id = int(x[1])
        rating = float(x[2])
        
        diff = rating - np.dot(W_rows[x_id % x_block_dim], H_rows[y_id % y_block_dim])
        err += diff ** 2
        n += 1
    return (err, n)

# Read every line from the file
def map_line(line):
    tokens = line.split(",")
    # parse the original data line, which is (row_id, column_id, value)
    return int(tokens[0]), int(tokens[1]), float(tokens[2])

# Convert the matrix to one dimensional array
def convertOneDim(matrix):
    res = []
    for i in range(0, len(matrix)):
        a = matrix[i]
        for j in range(0, len(a)):
            res.append(a[j])
    return res

if __name__ == '__main__':
    conf = pyspark.SparkConf().setAppName("Test")
    sc = pyspark.SparkContext(conf = conf)
    
    print ("Out-of-Core SGD Matrix Factorization begins....")
    csv_file = sys.argv[1]       # rating file
    K = int(sys.argv[2])         # rank
    W_path = sys.argv[3]         # W matrix file 
    H_path = sys.argv[4]         # H matrix file    
    N = 4                        # Node of every slave
    Node_num = 2                 # Num of node
    Partition_num = N * Node_num # Partition num
    num_iterations = 10
    eta = 0.001
    eta_decay = 0.99

    # blockify_data
    ratings = sc.textFile(csv_file, Partition_num).map(map_line).cache()
    max_x_id = ratings.map(lambda x: x[0]).max()
    max_y_id = ratings.map(lambda x: x[1]).max()
    print "max id (%d, %d)" % (max_x_id, max_y_id)
     
    # assume the id starts from 1
    x_block_dim = int((max_x_id + Partition_num) / Partition_num)
    y_block_dim = int((max_y_id + Partition_num) / Partition_num)

    # initialize_factor_matrices
    H_rdd = sc.parallelize(np.random.random((Partition_num, y_block_dim, K))).zipWithIndex().map(
            lambda (x, block_id): (block_id+1,x)).partitionBy(Partition_num)
    W_rdd = sc.parallelize(np.random.random((Partition_num, x_block_dim, K))).zipWithIndex().map(
            lambda (x, block_id): (block_id+1, x)).partitionBy(Partition_num)

#   t1 = time.clock()
    print("Start Stochastic Gradient Descent...")
    for iter in range(0, num_iterations):
        for i in range(0, Partition_num):
            # Select the rating blocks
            ratings_sub = ratings.filter(lambda x: ((x[0] / x_block_dim + i) % Partition_num)
                    == (x[1] / y_block_dim))
            ratings_block = ratings_sub.map(lambda x: (x[0] / x_block_dim + 1, x))
            # Re-sort the block id of H matrix based on the num i
            H_block = H_rdd.map(lambda (block_id, x): ((block_id + Partition_num - i - 1) % Partition_num + 1, x))
            # Group the rating block, H and W based on block id
            RWH_union = ratings_block.groupWith(W_rdd, H_block).partitionBy(Partition_num)
            # Update the H and W matrices using SGD, and return updated H and W
            sgd_updates = RWH_union.map(lambda (block_id, x): (block_id, sgd_on_one_block_func(
                eta, x_block_dim, y_block_dim, x[0], x[1], x[2]))).collect()
            W = []
            H = []
            # Update the value of H_rdd and W_rdd
            for updates in sgd_updates:
                block_id = updates[0]
                W.append([block_id, updates[1][0]])
                H.append([block_id, updates[1][1]])
            W_rdd = sc.parallelize(W).map(lambda x: (x[0], x[1])).partitionBy(Partition_num)
            H_rdd = sc.parallelize(H).map(lambda x: ((x[0] - 1 + i) % Partition_num + 1, 
                x[1])).partitionBy(Partition_num)
        eta *= eta_decay 
    
    # Sort the H and W matrices and store them into HDFS
    W = W_rdd.takeOrdered(Partition_num, key = lambda(block_id, x): block_id)
    W_array = convertOneDim(sc.parallelize(W).map(lambda (block_id, x): x).collect()) 
    H = H_rdd.takeOrdered(Partition_num, key = lambda (block_id, x): block_id)
    H_array = convertOneDim(sc.parallelize(H).map(lambda (block_id, x): x).collect())
    sc.parallelize(W_array).map(lambda x: ",".join(map(str,x))).coalesce(1).saveAsTextFile(W_path)
    sc.parallelize(H_array).map(lambda x: ",".join(map(str,x))).coalesce(1).saveAsTextFile(H_path)

# Evaluate the RMSE 
#    err = 0.0
#    n = 0
#    for i in range(0, N):
#        ratings_sub = ratings.filter(lambda x: ((x[0] / x_block_dim + i) % N) == (x[1] / y_block_dim))
#        ratings_block = ratings_sub.map(lambda x: (x[0] / x_block_dim + 1, x))
#        H_block = H_rdd.map(lambda (block_id, x): ((block_id + N - i - 1) % N + 1, x))
#        RWH_union = ratings_block.groupWith(W_rdd, H_block).partitionBy(N)
#        evaluate = RWH_union.map(lambda (block_id, x): evaluate_on_one_block_func(x_block_dim, y_block_dim, x[0], x[1], x[2])).collect()   
#        for i in range(0, len(evaluate)):
#            err += evaluate[i][0]
#            n += evaluate[i][1]
#    
#    t2 = time.clock()
#    print(iter, int(t2 - t1), err, np.sqrt(err / n))


