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



from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time #helper functions - see lstm.py

# load data
# - sliding window side: 50 elements
# - normalization
# - 90% for the training; 10% for testing 
X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', 50, True)

# ================= Create Recurrent Neural Network =================
print("Create recurrent neural network ...")
# RNN settings
# - network with 1-dimension input; 
# - 2 hidden layers of size: 20 and 100;
# - 1-dimension output

# initalize reecurrent neural network; 
model = Sequential()


model.add(LSTM(
    input_dim=1,
    output_dim=20,
    return_sequences=True))

model.add(LSTM(
    100,
    return_sequences=False))

model.add(Dense(output_dim=1))

model.add(Activation("linear")) # regression - predict value 

# compile neural network
model.compile(optimizer='rmsprop', loss='mse')

# ================= END of Create Recurrent Neural Network =================


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


# get the original value (no normalization) 
X_train_org, y_train_org, X_test_org, y_test_org = lstm.load_data('sp500.csv', 50, False)

predictions_org = []
for idx in range(y_test_org.shape[0]):
    #get the first value
    p0 = X_test_org[idx].astype(float)[0][0]
    pi = predictions[idx]
    predictions_org.append((pi+1)*p0)
    

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111) #111- 1x1 grid, first subplot
ax.plot(y_test_org, label='True Data')
plt.plot(predictions_org, label='Prediction')
plt.legend()
plt.title("RNN Result (Denormalized)")
plt.show()

#fig = plt.figure(facecolor='white')
#ax = fig.add_subplot(111) #111- 1x1 grid, first subplot
#ax.plot(y_test, label='True Data')
#plt.plot(predictions, label='Prediction')
#plt.legend()
#plt.title("RNN Result (Normalized)")
#plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(predictions_org, y_test_org.astype(float).tolist())

# up / down prediction accuracy?
updown_org = []
updown_pred = []
for idx in range(y_test_org.shape[0]):
    #get the last value
    last_element = X_test_org[idx].astype(float)[49][0]
    actual_value = y_test_org[idx].astype(float)

    pred_value = predictions_org[idx]
    updown_org.append(True if actual_value > last_element else False)
    updown_pred.append(True if pred_value > last_element else False)
                    
#XNOR
corrPredTrend = 0
for idx in range(len(updown_org)):
    if updown_org[idx] == updown_pred[idx]:
        corrPredTrend = corrPredTrend+1
#accuracy
accuracy = (corrPredTrend *1.0) / len(updown_org) * 100