#install.packages("rnn")
library(rnn)

set.seed(1234)

#set working directory
setwd("~/DS/GITHUB/MLP/stock_prediction")

dataset = read.csv("sp500.csv", col.names = "Y")

# Samples of0 time series
y = as.numeric(dataset$Y)
Y <- matrix(y, nrow = 50)

plot(as.vector(Y), col='blue', type='l', xlab="t", ylab = "value", main = "SP500 Index")


#standardize 
Y <- (Y - min(Y)) / (max(Y) - min(Y))
#transpose
Y <- t(Y)

train <- 1:60
test <- 61:84

#train the model
model <- trainr(Y = Y[train,],
                X = X[train,],
                learningrate = 0.05,
                hidden_dim = 16,
                numepochs = 1500)