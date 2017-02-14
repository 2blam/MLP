# deep learning / artifical neural network by 2blam

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import dataset
dataset = pd.read_csv("Speed Dating Data.csv")
dataset.shape #8378, 195

# preprocessing
# 1) extract related column first
# short form
# pf_o_xxx - partner’s stated preference at Time 1 
# XXX1_1 - what you look for in the opposite sex
# XXX2_1 - What do you think the opposite sex looks for in a date?
# XXX3_1 - How do you think you measure up (self evaluate)?
# Attractive - att
# Sincere - sin
# Intelligent - int
# Fun - fun
# Ambitious - amb
# Has shared interests/hobbies - sha

# note:


colnames = ["iid", "gender", "wave", "pid", "match", "samerace", "age_o", "race_o", \
"pf_o_att", "pf_o_sin", "pf_o_int", "pf_o_fun", "pf_o_amb", "pf_o_sha", \
"age", "field_cd", "race", "imprace", \
"imprelig", "goal", "date", "go_out", "career_c", \
"sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", \
"gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts", \
"music", "shopping", "yoga", "exphappy", \
"attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1", \
"attr2_1", "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1", \
"attr3_1", "sinc3_1", "intel3_1", "fun3_1"]

dataset = dataset[colnames] 

# 2) check valid data
# wave values 6 - 9 are in 1-10 scale, with different calculation, remove those rows
dataset = dataset[(dataset['wave']<6) | (dataset['wave'] > 9)] #6816 rows

# remove row with na value 
# check is there any na value in pid column
dataset["pid"].isnull().any()#is there any na value
dataset["pid"].isnull().sum()# how many - 10
dataset = dataset.dropna(subset = ['pid']) 

dataset["age_o"].isnull().any() #is there any na value
dataset["age_o"].isnull().sum() #how many - 89
dataset = dataset.dropna(subset = ['age_o'])

dataset["pf_o_att"].isnull().any() #is there any na value 
dataset["pf_o_att"].isnull().sum() #how many - 16
dataset = dataset.dropna(subset = ['pf_o_att'])

dataset["pf_o_fun"].isnull().any() #is there any na value 
dataset["pf_o_fun"].isnull().sum() #how many - 9
dataset = dataset.dropna(subset = ['pf_o_fun'])

dataset["pf_o_sha"].isnull().any() #is there any na value 
dataset["pf_o_sha"].isnull().sum() #how many -22
dataset = dataset.dropna(subset = ['pf_o_sha'])

dataset["age"].isnull().any() #is there any na value 
dataset["age"].isnull().sum() #how many -89
dataset = dataset.dropna(subset = ['age'])

dataset["field_cd"].isnull().any() #is there any na value 
dataset["field_cd"].isnull().sum() #how many - 18
dataset = dataset.dropna(subset = ['field_cd'])

dataset["imprace"].isnull().any() #is there any na value 
dataset["imprace"].isnull().sum() #how many - 16
dataset = dataset.dropna(subset = ['imprace'])

dataset["date"].isnull().any() #is there any na value 
dataset["date"].isnull().sum() #how many - 18
dataset = dataset.dropna(subset = ['date'])

dataset["career_c"].isnull().any() #is there any na value 
dataset["career_c"].isnull().sum() #how many - 38
dataset = dataset.dropna(subset = ['career_c'])

dataset["fun1_1"].isnull().any() #is there any na value 
dataset["fun1_1"].isnull().sum() #how many -9
dataset = dataset.dropna(subset = ['fun1_1'])

dataset["amb1_1"].isnull().any() #is there any na value 
dataset["amb1_1"].isnull().sum() #how many -9
dataset = dataset.dropna(subset = ['amb1_1'])

dataset["shar1_1"].isnull().any() #is there any na value 
dataset["shar1_1"].isnull().sum() #how many - 22
dataset = dataset.dropna(subset = ['shar1_1'])

dataset["attr3_1"].isnull().any() #is there any na value 
dataset["attr3_1"].isnull().sum() #how many - 26
dataset = dataset.dropna(subset = ['attr3_1'])

dataset.shape #6434, 57

# extract male and female data
dataset_male = dataset[(dataset["gender"] == 1)]
dataset_female = dataset[(dataset["gender"] == 0)]

for index, row in dataset_male.iterrows():
    print(row["pid"])



# encode categorical data 
# country
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# create dummy variables for country X X X
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column

# split into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0) # 25% for testing dataset

# re-scale feature values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 

# create neural network
import keras
from keras.models import Sequential
from keras.layers import Dense

# initalize neural network; 
# NOTE: more than 1 hidden layers, in theory, we call it deep neural network
classifier = Sequential()

# add the input layer & 1st hidden layer
classifier.add(Dense(output_dim=6,
                     init="uniform",
                     activation="relu",
                     input_dim=11)) #set input_dim for the 1st layer only

# add 2nd hidden layer
classifier.add(Dense(output_dim=6,
                     init="uniform",
                     activation="relu"))
                     
# add output layer
# 1 class (e.g. yes vs no): output_dim = 1 AND activation = sigmoid
# n classes (one hot encode): output_dim = n AND activation = softmax
classifier.add(Dense(output_dim=1,
                     init="uniform",
                     activation="sigmoid"))
# compile neural network
# adam - Adaptive Moment Estimation 
# binary_crossentropy - https://keras.io/objectives/
classifier.compile(optimizer="adam",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])

# fit training data to neural network
# batch_size - update the weight only after finish a batch of records
# epoch - 1 epoch is equal to the whole training set passed through the neural network
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# save the model
from keras.models import load_model
classifier.save("nn_model.h5")
# load the model
del classifier
classifier = load_model("nn_model.h5")

# predict
y_pred = classifier.predict(X_test) #probabilty
y_pred = (y_pred > 0.5)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = 1.0*(cm[0,0] + cm[1,1]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
