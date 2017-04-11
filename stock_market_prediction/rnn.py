# recurrent neural network by 2blam
# Reference: 
# - https://www.youtube.com/watch?v=ftMq5ps503w&feature=youtu.be
# - https://github.com/llSourcell/How-to-Predict-Stock-Prices-Easily-Demo
# - https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent

# import the libraries
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt
import pandas as pd


# ================= Create Recurrent Neural Network =================
print("Create recurrent neural network ...")
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time #helper libraries

# load_data(filename, seq_len, normalise_window)
# sp500.csv - 4170 values

X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', 50, True)
# 50 is the size of the sliding window
# 90% for the training; 10% for testing 

# get the inital value 
dataset = pd.read_csv('sp500.csv', header=None)
p0 = float(dataset.iloc[0])

# RNN settings
# initalize reecurrent neural network; 
model = Sequential()

# - network with 1-dimension input; 
# - 2 hidden layers of size: 50 and 100;
# - 1-dimension output

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))

model.add(LSTM(
    100,
    return_sequences=False))

# add output layer; output_dim =1
model.add(Dense(output_dim=1))

model.add(Activation("linear")) # as we do regression

# compile neural network
model.compile(optimizer='rmsprop', loss='mse')

# ================= END of Create Neural Network =================


# ================= Train the Neural Network =================
print("Train neural network ...")
# fit training data to recurrent neural network
# batch_size - update the weight only after finish a batch of records
# epoch - 1 epoch is equal to the whole training set passed through the neural network
#Train the model
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=1,
    validation_split=0.05)

# ================= END of Train the Neural Network =================

# save the model
#from keras.models import load_model
#classifier.save("model.h5")
## load the model
#del classifier
#classifier = load_model("model.h5")


# ================= Predict   =================
predictions = lstm.predict_point_by_point(model, X_test)

#denormalize
y_test_denormalized = [p0 * i for i in (y_test+1).tolist()]
predictions_denormalized = [p0 * i for i in (predictions+1).tolist()]

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y_test_denormalized, label='True Data')
plt.plot(predictions_denormalized, label='Prediction')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
mean_squared_error(predictions_denormalized, y_test_denormalized)

# up / down prediction accuracy?
truthResult = np.diff(y_test) > 0
predResult = np.diff(predictions) > 0
                    
#XNOR
corrPredTrend = ~(truthResult ^ predResult)

#accuracy
accuracy = (sum(corrPredTrend) *1.0) / len(corrPredTrend) * 100