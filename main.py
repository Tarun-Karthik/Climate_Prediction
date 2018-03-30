from keras.models import Sequential
from keras.layers import Dense
import numpy

dataset = numpy.loadtxt("Data_Jan.csv", delimiter=",")

Rainfall = dataset[...,3]

X = dataset[:,0:3]

print(Rainfall)
print(X)

#creating a model

model = Sequential()

model.add(Dense(6,input_dim=3,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='relu'))

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'])

model.fit(X,Rainfall, epochs=2000, batch_size=15)

scores = model.evaluate(X, Rainfall)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X)
print(predictions)
