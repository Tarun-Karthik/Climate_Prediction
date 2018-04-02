from keras.models import Sequential
from keras.layers import Dense
import numpy



month = [1,2,3,4,5,6,7,8,9,10,11,12]

model = Sequential()
model.add(Dense(5,input_dim=3,activation='sigmoid'))
model.add(Dense(5,activation='sigmoid'))
model.add(Dense(1,activation='linear'))

for x in month:
	dataset = numpy.loadtxt("./Training_Data/Month"+`x`+"/Train.csv", delimiter=",")
	output = dataset[:,3]
	input = dataset[:,0:3]
	model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['mean_absolute_error'])
	model.fit(input,output, epochs=30000, batch_size=30)
	scores = model.evaluate(input, output)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))	


for y in month:
	dataset2 = numpy.loadtxt("./Testing_Data/Month"+`y`+"/Test.csv", delimiter=",")
	Z = dataset2[:,0:3]
	predictions = model.predict(Z)
	numpy.savetxt("./Results/Month"+`y`+".csv",predictions)
