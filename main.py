from keras.models import Sequential
from keras.layers import Dense
import numpy

dataset = numpy.loadtxt("/Data/Data.csv", delimiter=",")
Max_Temp = dataset[...,0]
Min_Temp = dataset[...,1]
Rainfall = dataset[...,2]


#creating a model

model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

