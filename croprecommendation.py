# importing necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier,plot_tree

#loading dataset
dataset=pd.read_csv("Crop_recommendation.csv")

#dropping unnecessary columns
dataset=dataset.drop(columns=["Unnamed: 8","Unnamed: 9"])

#dataset exploration
print(type(dataset))
print(dataset.head())
print("Number of rows: ",len(dataset)," number of columns: ",len(dataset.columns))

#Checking for null data and removing it by deleting rows containing null data
print(dataset.isnull().sum())
dataset.dropna(inplace=True)

#Checking for duplicate data and removing it by deleting rows containing duplicate data and keeping first occurence
print(dataset.duplicated().sum())
dataset.drop_duplicates(keep="first",inplace=True)

#separating dataset into features and target
features=dataset.drop(["label"],axis="columns")
target=dataset["label"]
print(features)
print(target)

#scaling features
mms=MinMaxScaler()
nfeatures=mms.fit_transform(features.values)
print(nfeatures)

#splitting the data
x_train,x_test,y_train,y_test=train_test_split(nfeatures,target)

#training the model
model=DecisionTreeClassifier()
model.fit(x_train,y_train)

#evaluating the model
cm=confusion_matrix(y_test,model.predict(x_test))
cr=classification_report(y_test,model.predict(x_test))
print("Confusion Matrix: ",cm,"\n\nClassification report: ",cr,"\n\nAccuracy Score: ",accuracy_score(y_test,model.predict(x_test)))

#taking user values
ni=input("Enter nitrogen: ")
pho=input("Enter phosphorus: ")
pot=input("Enter potassium: ")
temp=input("Enter temperature: ")
hum=input("Enter humidity: ")
ph=input("Enter ph: ")
rain=input("Enter rainfall: ")

#Prediction
d=[[ni,pho,pot,temp,hum,ph,rain]]
nd=mms.transform(d)
ans=model.predict(nd)
print(ans)
