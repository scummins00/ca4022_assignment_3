#!/usr/bin/env python
# coding: utf-8

# # Gradient Boosted Tree
# The following Jupyter Notebook shows the use of Pyspark.ML `GBTRegressor`. In the notebook, we'll first define our schema structure and populate it with our data. Our data is currently stored in `cleaning.txt`, a simple text file representation of the output our 'Data Cleaning PIG Script' produces.
# 
# We'll perform some simple data transformations to ensure any categorical features are presented as DoubleType using Pyspark.ML's `StringIndexer` API. We'll do the same for any Integer typed features.
# 
# We'll then use the Pyspark.ML `VectorIndexer` to produce a vector representation of our input features. This step will be added as a stage in our ML Pipeline called **VectorAssembler**. Then we'll normalise the input values so that they are all on the same scale. This normalizer step will also be added to our Pipeline as a stage named **normalizer**.
# 
# Finally, we'll instantiate our GBT Model using Pyspark.ML's `GBTRegressor`. This will be the final stage in our ML Pipeline, and represents the `fit()` stage. It will simply be named **GBT**. We'll use our Pipeline to instantiate a model which will be fit on our training data. We'll then use this trained model to transform our test data and receive a salary prediction for each data point. We'll calculate a measure of success using **RMSE**.

# In[1]:


import findspark
findspark.init()


# In[2]:


import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession, Row
from pyspark import SparkConf
from pyspark.sql.functions import col
import pandas as pd
import numpy as np


# In[3]:


#Instantiate a spark session
spark = SparkSession     .builder     .appName("SparkML Pipeline Building")     .getOrCreate()


# In[4]:


# Let's define our schema
schema = StructType([    StructField("timestamp", StringType(), True),    StructField("company", StringType(), True),    StructField("level", StringType(), True),    StructField("title", StringType(), True),    StructField("totalyearlycompensation", IntegerType(), False),    StructField("location", StringType(), True),    StructField("yearsofexperience", FloatType(), False),    StructField("yearsatcompany", FloatType(), False),    StructField("tag", StringType(), True),    StructField("basesalary", IntegerType(), False),    StructField("stockgrantvalue", IntegerType(), False),    StructField("bonus", IntegerType(), False),    StructField("gender", StringType(), True),    StructField("cityid", StringType(), True),    StructField("dmaid", StringType(), True),    StructField("race", StringType(), True),    StructField("education", StringType(), True)])

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("csv")    .option("header", "false")    .option("delimiter", "\t")    .schema(schema)    .load("../data/replaced_salary_data/cleaned.txt")
data.show(n=5)


# ## Encoding of Categorical Features
# In the next cell, we will list all of our categorical features currently of the `string` type. All ML models require data to presented in numerical values. We will use the `StringIndexer` method to transform our categorical features into numerical columns.
# 
# This is performed as a pre-processing step to our pipeline.

# In[5]:


#Different indexer for each categorical column
cols_to_be_indexed = ['company', 'level', 'title', 'location', 'gender', 'race', 'education']

indexed_cols = ['company_index', 'level_index', 'title_index', 'location_index',
                'gender_index', 'race_index', 'education_index']

#Let's create a copy of our data to work from
indexed = data

indexer = StringIndexer(inputCols=cols_to_be_indexed, outputCols=indexed_cols)
indexed = indexer.fit(indexed).transform(indexed)


# ## Typecasting of Features
# In this step, we will convert all of our `integer` features into `double` features. Again, this is a formality, and is better practice for ML pipeline building. Like the previous step, we will transform our integer values straight away.

# In[6]:


#List of numerical columns to turn into double-type

numeric_cols = ['totalyearlycompensation', 'yearsofexperience', 'yearsatcompany', 'basesalary', 'stockgrantvalue', 'bonus',
               'cityid', 'dmaid']
for col in numeric_cols:
    indexed = indexed.withColumn(col, data[col].cast(DoubleType()))


# ## Assembling Input Features
# Here we assemble all of the input features for our model. This includes our previously Typecast features, and our Encoded categorical features.

# In[7]:


from pyspark.ml.feature import VectorAssembler

feature_list = ['gender_index', 'race_index', 'education_index', 'company_index', 'title_index',
                'level_index', 'location_index', 'totalyearlycompensation', 'yearsofexperience', 'yearsatcompany',
                'basesalary', 'stockgrantvalue', 'bonus', 'cityid', 'dmaid']

#This is our Vector Assembler object, it will be put into our Pipeline later on
vectorAssembler = VectorAssembler(inputCols=feature_list, outputCol='features', handleInvalid='keep')


# In[8]:


#Let's show the number of unique values per column
for col in feature_list:
    print(f"Distinct Count for column {col}: " + str(indexed.select(col).distinct().count()))


# ## Normalise Feature Set
# Our input features are not on one singular scale. This can damage the performance of our ML model. For example, `yearsofexperience` will be a small value, less than 50. Whereas `basesalary` can be upwards of 1,000,000. We require all values to be within a similar range.

# In[9]:


from pyspark.ml.feature import Normalizer

normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)


# ## Training, Test split
# In the following cell, we are simply splitting our data into training and test data.

# In[10]:


print(indexed.count())
print("Dropping na's")
indexed = indexed.dropna()
print(indexed.count())


# In[11]:


# Let's create our training and test data
splits = indexed.randomSplit([0.8, 0.2])
trainingData = splits[0]
testingData = splits[1]


# ## Pipeline Creation
# When following the previous steps from **Encoding of Categorical Features**, through to **Normalise Feature Set**, you'll notice that the `inputCol` used in the function is usually the `outputCol` from the previous function. This is done purposely so that the individual Transformer pieces of our pipeline fit together.
# 
# In our stages, we start with our Vector Assembler. That feeds into our Normalizer as the final transformation step. The final stage is the `fit()` stage which is the Gradient Boosted Trees Regressor model itself. We will define it in the following cells then instantiate our Pipeline.

# In[12]:


#Let's define our gbt with our target column as basesalary
gbt = GBTRegressor(labelCol='basesalary', featuresCol='features_norm', maxIter=100)


# In[13]:


#Now we instantiate our pipeline with the stages as discussed above.
pipeline = Pipeline(stages=[vectorAssembler, normalizer, gbt])

#We can then take this pipeline, fit it to our training data
model = pipeline.fit(trainingData)


# ## Prediction on Test Data
# Now that we've built our model, let's test its accuracy using the test data we set aside earlier. We can apply our model to data using `model.transform(<data>)`. This will create a DataFrame object, where each row has a new column named `prediction`. 
# 
# Let's take a look at the prediction column. As our dataset is quite large, we'll view a subset of the columns.

# In[14]:


prediction = model.transform(trainingData)

#Let's just quickly make our predictions rounded to 2 decimals
prediction = prediction.withColumn("prediction",round(prediction["prediction"],2))

prediction.select('company', 'location', 'level', 'yearsofexperience',
                  'gender', 'education', 'race', 'basesalary', 'prediction').show(n=10)


# In[15]:


evaluator = RegressionEvaluator(
    labelCol="basesalary", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# ## Model Tuning
# In the following section, we'll be looking at using `CrossValidator` to improve the overall performance of model. We'll also be seeting up a Grid Search problem using `ParamGridBuilder` from Pyspark.ML for hyperparameter tuning. These are important steps in model development.
# 
# **Important Note:** Running a Grid Search on a CPU will take considerable time to finish, and is generally not advisable.

# In[16]:


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# In[ ]:


#Running this on a CPU is not advisable.

#Let's set up our grid search
paramGrid = ParamGridBuilder()     .addGrid(gbt.maxIter, [20, 50, 100, 200, 250])     .addGrid(gbt.maxDepth, [0, 1, 2, 5, 10, 20 , 30])     .addGrid(gbt.stepSize, [0.01, 0.05, 0.1])     .addGrid(gbt.subsamplingRate, [0.05,0.1, 0.2, 0.5, 1.0])     .build()

#Let's set up our cross validator
crossval = CrossValidator(estimator=pipeline,
                         estimatorParamMaps = paramGrid,
                         evaluator = evaluator,
                         numFolds=5)

#Fit the model
cvModel = crossval.fit(trainingData)

