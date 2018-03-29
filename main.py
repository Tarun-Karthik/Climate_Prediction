from keras.models import Sequential
from keras.layers import Dense
import numpy

dataset = numpy.loadtxt("/Data/Data.csv", delimiter=",")
Max_Temp = dataset[...,0]
Min_Temp = dataset[...,1]
Rainfall = dataset[...,2]


#creating a model

model = Sequential()
model.add(Dense(2, input_dim=2, activation='linear'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

