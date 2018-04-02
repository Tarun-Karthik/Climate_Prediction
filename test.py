import numpy
input = numpy.loadtxt("./Testing_Data/Month1/Test.csv", delimiter=",")
data = input[:,0:3]
print(data)
input2 = numpy.loadtxt("./Testing_Data/Month2/Test.csv", delimiter=",")
data = numpy.vstack((data,input2[:,0:3]))
print(data)
