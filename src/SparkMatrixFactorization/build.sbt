name := "Spark Matrix Factorization"
 
version := "1.0"
 
scalaVersion := "2.10.5"
 
libraryDependencies += "org.apache.spark" % "spark-core_2.10" % "1.6.0"

javaOptions in (Test, run) += "-Xmx8G"
