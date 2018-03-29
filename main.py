from keras.models import Sequential
from keras.layers import Dense
import numpy

dataset = numpy.loadtxt("Data_Jan.csv", delimiter=",")

Rainfall = dataset[...,2]

X = dataset[:,0:2]

#creating a model

model = Sequential()

model.add(Dense(4,input_dim=2,activation='sigmoid'))
model.add(Dense(1,activation='linear'))

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_absolute_error'])

model.fit(X,Rainfall, epochs=100, batch_size=15)

scores = model.evaluate(X ,Rainfall)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))
