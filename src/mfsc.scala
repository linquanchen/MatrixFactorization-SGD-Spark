
import java.util.Random

import scala.collection.mutable.Map
import scala.collection.mutable.Set
import Array._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer
import scala.math
import scala.collection._

import org.apache.spark.{SparkConf, SparkContext}

def dotProduct[T <% Float](as: List[T], bs: List[T]) = {
   require(as.size == bs.size)
   (for ((a, b) <- as zip bs) yield a * b) sum
}

// Using SGD to update the W and H matrices
def sgd_on_one_block_func(step_size : Float, x_block_dim: Int, y_block_dim: Int, R_rows: Iterable[(Int, Int, Float)], W_rows_in: Iterable[Array[Array[Float]]], H_rows_in: Iterable[Array[Array[Float]]]): Array[Array[Array[Float]]] = {
	val W_rows: Array[Array[Float]] = W_rows_in.toList(0)
	val H_rows: Array[Array[Float]] = H_rows_in.toList(0)
	val result = new ArrayBuffer[Array[Array[Float]]]
	// Iterate every block
	for (x <- R_rows) {
		val x_id = x._1.toInt
		val y_id = x._2.toInt
		val rating = x._3.toFloat

		val diff = rating - dotProduct(W_rows(x_id % x_block_dim).toList, H_rows(y_id % y_block_dim).toList)
		val W_gradient = H_rows(y_id % y_block_dim).map(x => x * diff * (-2))
		// val W_gradient = H_rows(y_id % y_block_dim):* diff :* (-2)
		// W_rows(x_id % x_block_dim) = W_rows(x_id % x_block_dim) :- step_size * W_gradient
		for (i <- 0 to W_rows(x_id % x_block_dim).length - 1) {
			W_rows(x_id % x_block_dim)(i) =  W_rows(x_id % x_block_dim)(i) - step_size * W_gradient(i)
		}

		val H_gradient = W_rows(x_id % x_block_dim).map(x => x * diff * (-2))
		// H_rows(y_id % y_block_dim) = H_rows(y_id % y_block_dim) :- step_size * H_gradient
		for (i <- 0 to H_rows(y_id % y_block_dim).length - 1) {
			H_rows(y_id % y_block_dim)(i) = H_rows(y_id % y_block_dim)(i) - step_size * H_gradient(i)
		}
	}
	result += W_rows
	result += H_rows
	result.toArray
}

// Evalute the RMSE of blocks in diagonal
def evaluate_on_one_block_func(x_block_dim: Int, y_block_dim: Int, R_rows: Array[(Int, Int, Float)], W_rows: Array[Float], H_rows: Array[Float]): (Double, Int) = {
	val err = 0.0
	val n = 0
	val W_rows = W_rows(0).toList
	val H_rows = H_rows(0).toList 
	for (x <- R_rows) {
		val x_id = x._1.toInt
		val y_id = x._2.toInt
		val rating = x._3.toFloat
		val diff = rating - dotProduct(W_rows(x_id % x_block_dim), H_rows(y_id % y_block_dim))
		err += diff * diff
		n += 1
	} 
	(err, n)
}

// Convert the matrix to one dimensional array
def convertOneDim(matrix: Array[Array[Array[Float]]]): Array[List[Float]] = {
	val l = matrix.length
	val w = matrix(0).length
	val res = new ArrayBuffer[List[Float]]
	for (i <- 0 to l - 1) {
		val a = matrix(i)
		for (j <- 0 to w - 1) {
			res += a(j).toList
		}
	}
	res.toArray
}

def randomArrayCreation(Partition_num: Int, y_block_dim: Int, K: Int): Array[Array[Array[Float]]] = {
	val r = new scala.util.Random
	val matrix = ofDim[Float](Partition_num,y_block_dim,K)
	for (i <- 0 to Partition_num - 1) {
		for (j <- 0 to y_block_dim - 1) {
			for (k <- 0 to K - 1) {
				matrix(i)(j)(k) = r.nextFloat
			} 
		}
	}
	matrix
}

// Read every line from the file
def map_line(line: String): (Int, Int, Float) = {
	val tokens = line.split(",")
	(tokens(0).toInt, tokens(1).toInt, tokens(2).toFloat)
}


def main(args: Array[String]) {
val
	val conf = new SparkConf().setAppName("Test")
	val sc = new SparkContext(conf)

	println("Out-of-Core SGD Matrix Factorization begins....")

	val csv_file = args(1)		// rating file
	val K = args(2).toInt       // rank
	val W_path = args(3)		// W matrix file 
	val H_path = args(4)		// H matrix file  
	val N = 4					// Node of every slave
	val Node_num = 2 			// Num of node
	val Partition_num = N * Node_num    //Partition num
	val num_iterations = 10
	var eta : Float = 0.001.toFloat
	val eta_decay : Float = 0.99.toFloat

	// blockify_data
	// val ratings = sc.textFile("/input/ratings_1M.csv", Partition_num).map(line => map_line(line)).cache()    //tbd
	val inputFile = sc.textFile("/input/ratings_1M.csv")
	val ratings = inputFile.map(line => map_line(line)).cache()

	val max_x_id = ratings.map(x => x._1).max()
	val max_y_id = ratings.map(x => x._2).max()
	printf("max id (%d, %d)", max_x_id, max_y_id)

	//assume the id starts from 1
	val x_block_dim = (max_x_id + Partition_num) / Partition_num
	val y_block_dim = (max_y_id + Partition_num) / Partition_num

    //initialize_factor_matrices

 //    val H_rdd = sc.parallelize(randomArrayCreation(n_y)).zipWithIndex().map{ case (x, block_id) => (block_id + 1, x)}.partitionBy(Partition_num)
	// val W_rdd = sc.parallelize(randomArrayCreation(n_x)).zipWithIndex().map{ case (x, block_id) => (block_id + 1, x)}.partitionBy(Partition_num)

	var H_rdd = sc.parallelize(randomArrayCreation(Partition_num, y_block_dim, K)).zipWithIndex().map{ case (x, block_id) => ((block_id + 1).toInt, x)}
	var W_rdd = sc.parallelize(randomArrayCreation(Partition_num, x_block_dim, K)).zipWithIndex().map{ case (x, block_id) => ((block_id + 1).toInt, x)}

//  t1 = Calendar.getInstance().getTime()
    print("Start Stochastic Gradient Descent...")
    for (iter <- 0 to num_iterations - 1) {
    	for (i <- 0 to Partition_num - 1) {
			// Select the rating blocks
			val ratings_sub = ratings.filter(x => (x._1 / x_block_dim + i) % Partition_num == x._2 / y_block_dim)
			val ratings_block = ratings_sub.map(x => (x._1 / x_block_dim + 1, x) )
			// Re-sort the block id of H matrix based on the num i
			val H_block = H_rdd.map{ case(block_id, x) => ((block_id + Partition_num - i - 1) % Partition_num + 1, x)}
			// Group the rating block, H and W based on block id
            // val RWH_union = ratings_block.groupWith(W_rdd, H_block).partitionBy(Partition_num)
			val RWH_union = ratings_block.groupWith(W_rdd, H_block)

            // Update the H and W matrices using SGD, and return updated H and W
            val sgd_updates = RWH_union.map{ case(block_id, x) => (block_id, sgd_on_one_block_func(eta, x_block_dim, y_block_dim, x._1, x._2, x._3))}.collect()

		    var W = new ArrayBuffer[(Int, Array[Array[Float]])]
		    var H = new ArrayBuffer[(Int, Array[Array[Float]])]
            // Update the value of H_rdd and W_rdd
            for (updates <- sgd_updates) {
            	val block_id = updates._1
            	W.append((block_id, updates._2(0)))
            	H.append((block_id, updates._2(1)))
            }
            W_rdd = sc.parallelize(W).map(x => (x._1, x._2))
            H_rdd = sc.parallelize(H).map(x => ((x._1 - 1 + i) % Partition_num + 1, x._2))
            // W_rdd = sc.parallelize(W).map(x => (x._1, x._2)).partitionBy(Partition_num)
            // H_rdd = sc.parallelize(H).map(x => ((x._1 - 1 + i) % Partition_num + 1, x._2)).partitionBy(Partition_num)
    	}
    	eta = eta * eta_decay 
    }
    

	// Sort the H and W matrices and store them into HDFS
    val W_sorted = W_rdd.takeOrdered(Partition_num)(Ordering[Int].on(x => x._1))
    val W_array = convertOneDim(sc.parallelize(W_sorted).map{case (block_id, x) => x}.collect()) 
    val H_sorted = H_rdd.takeOrdered(Partition_num)(Ordering[Int].on(x => x._1))
    val H_array = convertOneDim(sc.parallelize(H_sorted).map{case (block_id, x) => x}.collect())

    sc.parallelize(W_array).map(x => x.mkString(",")).coalesce(1).saveAsTextFile("/output/W.csv")
    sc.parallelize(H_array).map(x => x.mkString(",")).coalesce(1).saveAsTextFile("/output/H.csv")

//Evaluate the RMSE 
   	// val err = 0.0
   	// val n = 0
   	// for (i <- 0 to N -1 ) {
   	// 	ratings_sub = ratings.filter((_._1 / x_block_dim + i) % N == (_._2 / y_block_dim))
    //    	ratings_block = ratings_sub.map(x => (x._1 / x_block_dim + 1, x) )
    //    	H_block = H_rdd.map{ case(block_id, x) => ((block_id + N - i - 1) % N + 1, x)}
    //    	RWH_union = ratings_block.groupWith(W_rdd, H_block).partitionBy(N)
    //    	val evaluate = RWH_union.map{ case (block_id, x) => evaluate_on_one_block_func(x_block_dim, y_block_dim, x._1, x._2, x._3)}.collect()  
    //    	for (i <- 0 to evaluate.length - 1) {
    //    		err += evaluate[i]._1
    //    		n += evaluate[i]._1
    //    	}
   	// }
    // val t2 = Calendar.getInstance().getTime()	
    // print("iter: %d, t2 - t1: %d, err: %f, sqrt(err): %09.5f", iter, int(t2 - t1), err, math.sqrt(err.toDouble))

}


	