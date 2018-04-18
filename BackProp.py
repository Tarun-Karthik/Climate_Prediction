from keras.models import Sequential
from keras.layers import Dense
import numpy

months = [1,2,3,4,5,6,7,8,9,10,11,12]
Data = []

model = Sequential()
model.add(Dense(10,input_dim=3,activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(1,activation='linear'))

for x in months:
    dataset = numpy.loadtxt("./Training_Data/Month"+`x`+"/Train.csv", delimiter=",")
    Data.append(dataset)

Data = numpy.vstack(Data)
input = Data[:,0:3]
output = Data[:,3]

model.compile(optimizer='nadam',loss='mean_absolute_error', metrics=['mean_absolute_error'])
model.fit(input,output, epochs=5000, batch_size=45)
scores = model.evaluate(input, output)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))


for y in months:
    dataset = numpy.loadtxt("./Testing_Data/Month"+`y`+"/Test.csv", delimiter=",")
    input = dataset[:,0:3]
    predictions = model.predict(input)
    numpy.savetxt("./Results/Month"+`y`+".csv",numpy.hstack((input,predictions)),delimiter = ',',fmt='%f')
