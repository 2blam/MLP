setwd("~/Documents/GitHub/MLP/speed_dating_experiment")

dataset = read.csv("Speed Dating Data.csv")

dim(dataset) #8378 x 195 

colnames(dataset)
unique(dataset$id)
