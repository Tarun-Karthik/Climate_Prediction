from keras.models import Sequential
from keras.layers import Dense
import numpy

dataset = numpy.loadtxt("Data_Jan.csv", delimiter=",")
Max_Temp = dataset[...,0]
Min_Temp = dataset[...,1]
Rainfall = dataset[...,2]

X = [Max_Temp,Min_Temp]

#creating a model

model = Sequential()
model.add(Dense(1, input_dim=2, activation='linear'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='linear'))


model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

model.fit(X,Rainfall, epochs=12, batch_size=15)

scores = model.evaluate(X ,Rainfall)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

