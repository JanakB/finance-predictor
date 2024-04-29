#Importing Required Libraries
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#Loading Data
col_names = ['Pragnency', 'Glucose', 'Blod Pressure', 'Skin Thikness', 'Insulin', 'BMI', 'DFP', 'Age', 'Diabetes']
# load dataset
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)

pima.head()

#Feature Selection
#Here, you need to divide given columns into two types of variables dependent(or target variable) and independent variable(or feature variables).
#split dataset in features and target variable
# feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
# X = pima[feature_cols] # Features
# y = pima.label # Target variable


#split dataset in features and target variable
feature_cols = ['Pragnency', 'Glucose', 'Blod Pressure', 'Skin Thikness', 'Insulin', 'BMI', 'DFP', 'Age', 'Diabetes']
X = pima[feature_cols] # Features
y = pima.label # Target variable


#Splitting Data
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#Building Decision Tree Model
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


#Evaluating the Model
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Visualizing Decision Trees

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import six

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


#Optimizing Decision Tree Performance
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
