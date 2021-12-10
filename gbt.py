import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

#Instantiate a spark session
spark = SparkSession \
    .builder \
    .appName("SparkML GBT") \
    .getOrCreate()

# Let's define our schema
schema = StructType([\
    StructField("date", DateType(), True),\
    StructField("time", StringType(), True),\
    StructField("company", StringType(), True),\
    StructField("level", StringType(), True),\
    StructField("title", StringType(), True),\
    StructField("totalyearlycompensation", IntegerType(), False),\
    StructField("location", StringType(), True),\
    StructField("yearsofexperience", FloatType(), False),\
    StructField("yearsatcompany", FloatType(), False),\
    StructField("tag", StringType(), True),\
    StructField("basesalary", IntegerType(), False),\
    StructField("stockgrantvalue", IntegerType(), False),\
    StructField("bonus", IntegerType(), False),\
    StructField("gender", StringType(), True),\
    StructField("cityid", StringType(), True),\
    StructField("dmaid", StringType(), True),\
    StructField("race", StringType(), True),\
    StructField("education", StringType(), True)])

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("csv")\
    .option("header", "false")\
    .option("delimiter", "\t")\
    .schema(schema)\
    .load("data/seperated_time_data/cleaned.txt")

#Different indexer for each categorical column
cols_to_be_indexed = ['company', 'level', 'title', 'location', 'gender', 'race', 'education']

indexed_cols = ['company_index', 'level_index', 'title_index', 'location_index',
                'gender_index', 'race_index', 'education_index']

#Let's create a copy of our data to work from
indexed = data

indexer = StringIndexer(inputCols=cols_to_be_indexed, outputCols=indexed_cols)
indexed = indexer.fit(indexed).transform(indexed)

#List of numerical columns to turn into double-type

numeric_cols = ['totalyearlycompensation', 'yearsofexperience', 'yearsatcompany', 'basesalary', 'stockgrantvalue', 'bonus', 'cityid', 'dmaid']
for col in numeric_cols:
    indexed = indexed.withColumn(col, data[col].cast(DoubleType()))


feature_list = ['gender_index', 'race_index', 'education_index', 'company_index', 'title_index',
                'level_index', 'location_index', 'totalyearlycompensation', 'yearsofexperience', 'yearsatcompany',
                'basesalary', 'stockgrantvalue', 'bonus', 'cityid', 'dmaid']

#This is our Vector Assembler object, it will be put into our Pipeline later on
vectorAssembler = VectorAssembler(inputCols=feature_list, outputCol='features', handleInvalid='keep')

normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

#Drop nan rows
indexed = indexed.dropna()

# Let's create our training and test data
splits = indexed.randomSplit([0.8, 0.2])
trainingData = splits[0]
testingData = splits[1]

#Let's define our gbt with our target column as basesalary
gbt = GBTRegressor(labelCol='basesalary', featuresCol='features_norm', maxIter=100)

#Now we instantiate our pipeline with the stages as discussed above.
pipeline = Pipeline(stages=[vectorAssembler, normalizer, gbt])

#We instantiate our validator
evaluator = RegressionEvaluator(
    labelCol="basesalary", predictionCol="prediction", metricName="rmse")

#Let's set up our grid search (Running on CPU is not advisable)
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [20, 50, 100, 200, 250]) \
    .addGrid(gbt.maxDepth, [0, 1, 2, 5, 10, 20 , 30]) \
    .addGrid(gbt.stepSize, [0.01, 0.05, 0.1]) \
    .addGrid(gbt.subsamplingRate, [0.05,0.1, 0.2, 0.5, 1.0]) \
    .build()

#Let's set up our cross validator
crossval = CrossValidator(estimator=pipeline,
                         estimatorParamMaps = paramGrid,
                         evaluator = evaluator,
                         numFolds=5)

#Fit the model to CV
cvModel = crossval.fit(trainingData)

#Evaluate on unseen data
evaluator.evaluate(cvModel.transform(testingData))

#Let's save our model
path = tempfile.mkdtemp()
model_path = path + "/model"
cvModel.write().save(model_path)