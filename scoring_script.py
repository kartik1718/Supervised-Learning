import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")
from pandas_profiling import ProfileReport
import datetime as DT
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import statistics
from time import clock
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold


##Loading the model file
loaded_model = pickle.load(open('Best_classifier.pckl', 'rb'))
#Read Test set file
data_read_test=pd.read_csv("test_set.csv")
#See the basic stats of the model
data_read_test.head()
data_read_test = data_read_test.drop(data_read_test.columns[0],axis='columns') 
data_read_test.info() 
data_read_test.describe()

#optional profiling
profile = ProfileReport(data_read_test, title="Pandas Profiling Report")
#profile

#
data_read_test=data_read_test.drop('X32',axis='columns') 

#Getting the shape of dataframe
print("shape of the data:", data_read_test.shape)

#We don't need to encode into numerical as all the columns are numerical 
X = data_read_test

#Checking the variance again in the dataset
sel_variance_threshold = VarianceThreshold() 
X_train_remove_variance = sel_variance_threshold.fit_transform(X)
print(X_train_remove_variance.shape)
del X_train_remove_variance

result = loaded_model.predict(data_read_test)#Predicting the test set
data_read_test['Y'] = pd.DataFrame(result)

#Dumping the predictions to a csv file
data_read_test.to_csv('test_predictions.csv')

