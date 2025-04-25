cols_to_drop = ['ID','Start_Time','End_Time','Description','Street',
                'Weather_Timestamp','Zipcode','County','City','Airport_Code','Precipitation(in)'] 


df = df.drop(*cols_to_drop)


# Replace nan with Nulls
columns = df.dtypes
for cols, typ in columns:
    if typ != 'boolean':
        df = df.withColumn(cols,when(isnan(col(cols)),None).otherwise(col(cols)))


label = 'Severity'
string_cols =  [cols[0] for cols in df.dtypes if cols[1] == "string" ]
num_cols = [cols[0] for cols in df.dtypes if cols[1] == "int" or cols[1] == "double" ]
num_cols.remove(label)
bool_cols = [cols[0] for cols in df.dtypes if cols[1] == "boolean"]


df = df.fillna("unknown",string_cols)
df = df.fillna(0,num_cols)

Dict_Null3 = {col:df.filter(isnull(df[col[0]])).count() for col in df.dtypes }
Dict_Null3

for c in bool_cols:
    df = df.withColumn(c,col(c).cast("integer"))    


(train, test) = df.randomSplit([0.7, 0.3])
train.count(), test.count()


from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sIndexer = [StringIndexer(inputCol=cols, outputCol=cols+"Index") for cols in  string_cols]


assembler = VectorAssembler(inputCols=[s.getOutputCol() for s in sIndexer]+bool_cols+num_cols, outputCol='features')

lr = LogisticRegression(featuresCol="features", labelCol=label)


pipeline = Pipeline(stages=sIndexer+[assembler, lr])
model = pipeline.fit(train)

preds = model.transform(train)
print("Prediction")
preds.select("Severity","prediction").show(20)

# evaluate the accuracy of the model using the test set
evaluator = MulticlassClassificationEvaluator(metricName='accuracy', labelCol="Severity")
accuracy = evaluator.evaluate(preds)

print()
print('#####################################')
print(f"Accuracy is {accuracy}")
print('#####################################')
print()

from onnxmltools import convert_sparkml
from onnxmltools.convert.sparkml.utils import buildInitialTypesSimple
from onnxmltools.convert.common.data_types import StringTensorType, FloatTensorType
import mlflow.onnx

import os
if not os.path.exists('model_onnx/'):
    os.mkdir('model_onnx/')
    
if not os.path.exists('model_spark/'):
    os.mkdir('model_spark/')
    
if not os.path.exists('model_mlflow'):
    os.mkdir('model_mlflow')




# Native ONNX framework
initial_types = buildInitialTypesSimple(df.drop('Severity'))
# initial_types
onnx_model = convert_sparkml(model, 'Pyspark model', initial_types, spark_session = spark)


with open(os.path.join("model_onnx/", "model.onnx"), "wb") as f:
    f.write(onnx_model.SerializeToString())


model.save('model_spark/spark_model')

mlflow.onnx.save_model(onnx_model,"model_mlflow/mlflow_model")

