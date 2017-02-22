# deep learning / artifical neural network by 2blam

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import dataset
dataset = pd.read_csv("Speed Dating Data.csv")
dataset.shape #8378, 195

# ================= Data Processing =================

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


# 1) drop irrelvant columns
colnames = ["iid", "gender", "wave", "pid", "match", \
"age", "field_cd", "race", "imprace", \
"imprelig", "goal", "date", "go_out", "career_c", \
"sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", \
"gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts", \
"music", "shopping", "yoga", "exphappy", \
"attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1", \
"attr2_1", "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1", \
"attr3_1", "sinc3_1", "intel3_1", "fun3_1"]

dataset = dataset[colnames] 

# 2) drop data in different scale
# wave values 6 - 9 are in 1-10 scale, with different calculation, remove those rows
dataset = dataset[(dataset['wave']<6) | (dataset['wave'] > 9)] #6816 rows

# 3) remove rows with missing values
idx = dataset.isnull().any(1).nonzero()[0]
print (len(idx))
dataset = dataset.drop(dataset.index[idx]) # drop rows with na values

# 4) pair the male and female member to form a new dataset
# change iid and pid as int type
dataset["iid"] = dataset["iid"].astype(int)
dataset["pid"] = dataset["pid"].astype(int)

# extract male and female data
dataset_male = dataset[(dataset["gender"] == 1)]
dataset_male = dataset_male.drop_duplicates()
dataset_female = dataset[(dataset["gender"] == 0)]
dataset_female = dataset_female.drop_duplicates()

# rename female dataset columns
dataset_female.columns = [i + "_F" for i in colnames ]

dataset_final = pd.DataFrame()

# merge dataset
for idx in dataset_male.index.values:
    male_info = dataset.loc[idx, :]
    male_iid = male_info["iid"].astype(int)
    female_pid = male_info["pid"].astype(int)
    rowIdx = ((dataset_female.iid_F == female_pid) & (dataset_female.pid_F == male_iid))
    
    # if exists
    if (sum(rowIdx) > 0):
        rowIdx2 = dataset_female[rowIdx].index[0]
        female_info = dataset_female.loc[rowIdx2, :]
        
        # pair up male and female
        combined_row = male_info.append(female_info)
        # append to dataset_final
        dataset_final = dataset_final.append(combined_row, ignore_index=True)

# drop irrevlant columns
drop_colnames = ["iid", "iid_F", "wave", "wave_F", "match_F", "pid", "pid_F", "gender", "gender_F"]
dataset_final = dataset_final.drop(labels=drop_colnames, axis=1)

dataset_final.shape #3167, 87

# encode categorical data 
# field_cd, field_cd_F, race, race_F, goal, goal_F, date, date_F, go_out, go_out_F, career_c, career_c_F
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# create dummy variables for career_c
colIdx = dataset_final.columns.get_loc("career_c")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['career_c'+str(i) for i in range(1, col_data.shape[1]+1)]
career_c_data = pd.DataFrame(col_data, columns=new_colnames)

# create dummy variables for career_c_F
colIdx = dataset_final.columns.get_loc("career_c_F")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['career_c_F'+str(i) for i in range(1, col_data.shape[1]+1)]
career_c_F_data = pd.DataFrame(col_data, columns=new_colnames)
            
# create dummy variables for date
colIdx = dataset_final.columns.get_loc("date")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['date'+str(i) for i in range(1, col_data.shape[1]+1)]
date_data = pd.DataFrame(col_data, columns=new_colnames)
            
# create dummy variables for date_F
colIdx = dataset_final.columns.get_loc("date_F")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['date_F'+str(i) for i in range(1, col_data.shape[1]+1)]
date_F_data = pd.DataFrame(col_data, columns=new_colnames)

# create dummy variables for field_cd
colIdx = dataset_final.columns.get_loc("field_cd")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['field_cd'+str(i) for i in range(1, col_data.shape[1]+1)]
field_cd_data = pd.DataFrame(col_data, columns=new_colnames)
 
# create dummy variables for field_cd_F
colIdx = dataset_final.columns.get_loc("field_cd_F")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['field_cd_F'+str(i) for i in range(1, col_data.shape[1]+1)]
field_cd_F_data = pd.DataFrame(col_data, columns=new_colnames)

# create dummy variables for go_out
colIdx = dataset_final.columns.get_loc("go_out")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['go_out'+str(i) for i in range(1, col_data.shape[1]+1)]
go_out_data = pd.DataFrame(col_data, columns=new_colnames)

# create dummy variables for go_out_F
colIdx = dataset_final.columns.get_loc("go_out_F")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['go_out_F'+str(i) for i in range(1, col_data.shape[1]+1)]
go_out_F_data = pd.DataFrame(col_data, columns=new_colnames)

# create dummy variables for goal
colIdx = dataset_final.columns.get_loc("goal")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['goal'+str(i) for i in range(1, col_data.shape[1]+1)]
goal_data = pd.DataFrame(col_data, columns=new_colnames)

# create dummy variables for go_out_F
colIdx = dataset_final.columns.get_loc("goal_F")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['goal_F'+str(i) for i in range(1, col_data.shape[1]+1)]
goal_F_data = pd.DataFrame(col_data, columns=new_colnames)

# create dummy variables for race
colIdx = dataset_final.columns.get_loc("race")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['race'+str(i) for i in range(1, col_data.shape[1]+1)]
race_data = pd.DataFrame(col_data, columns=new_colnames)

# create dummy variables for race_F
colIdx = dataset_final.columns.get_loc("race_F")
col_data = dataset_final.iloc[:, colIdx:(colIdx+1)]
onehotencoder = OneHotEncoder(categorical_features=[0])
col_data = onehotencoder.fit_transform(col_data).toarray()
col_data = col_data[:, 1:] # avoid dummy variable trap, remove 1 dummy variable column
new_colnames = ['race_F'+str(i) for i in range(1, col_data.shape[1]+1)]
race_F_data = pd.DataFrame(col_data, columns=new_colnames)

#column bind dataset_final
dataset_final = pd.concat([dataset_final, career_c_data], axis=1)
dataset_final = pd.concat([dataset_final, career_c_F_data], axis=1)
dataset_final = pd.concat([dataset_final, date_data], axis=1)
dataset_final = pd.concat([dataset_final, date_F_data], axis=1)
dataset_final = pd.concat([dataset_final, field_cd_data], axis=1)
dataset_final = pd.concat([dataset_final, field_cd_F_data], axis=1)
dataset_final = pd.concat([dataset_final, go_out_data], axis=1)
dataset_final = pd.concat([dataset_final, go_out_F_data], axis=1)
dataset_final = pd.concat([dataset_final, goal_data], axis=1)
dataset_final = pd.concat([dataset_final, goal_F_data], axis=1)
dataset_final = pd.concat([dataset_final, race_data], axis=1)
dataset_final = pd.concat([dataset_final, race_F_data], axis=1)

# ================= END of Data Processing =================


# ================= Prepare training and test data =================
X = dataset_final.drop(labels=["match", "career_c", "career_c_F", "date", \
                               "date_F", "field_cd", "field_cd_F", "go_out", \
                               "go_out_F", "goal", "goal_F", "race", "race_F"], axis=1)
X.shape

y = dataset_final["match"].astype(int)

# split into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0) # 20% for testing dataset

# re-scale feature values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 

# ================= END of Prepare training and test data =================



# ================= Create Neural Network =================

import keras
from keras.models import Sequential
from keras.layers import Dense

# NN settings
numOfInputNodes = 172
numOfHiddenNodes = 86
numOfEpoch = 100

# initalize neural network; 
# NOTE: more than 1 hidden layers, in theory, we call it deep neural network
classifier = Sequential()

# add the input layer & 1st hidden layer
classifier.add(Dense(output_dim=numOfHiddenNodes,
                     init="uniform",
                     activation="relu",
                     input_dim=numOfInputNodes)) #set input_dim for the 1st layer only

# add 2nd hidden layer
classifier.add(Dense(output_dim=numOfHiddenNodes,
                     init="uniform",
                    activation="relu"))


# add 3rd hidden layer
classifier.add(Dense(output_dim=numOfHiddenNodes,
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

# ================= END of Create Neural Network =================


# ================= Train the Neural Network =================
# fit training data to neural network
# batch_size - update the weight only after finish a batch of records
# epoch - 1 epoch is equal to the whole training set passed through the neural network
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=numOfEpoch)

# ================= END of Train the Neural Network =================

# save the model
#from keras.models import load_model
#classifier.save("model.h5")
## load the model
#del classifier
#classifier = load_model("model.h5")


# ================= Predict and calculate the accuracy =================
y_pred = classifier.predict(X_test) #probabilty
y_pred = (y_pred > 0.5)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = 1.0*(cm[0,0] + cm[1,1]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print(accuracy * 100)