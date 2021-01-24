import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class Network(object):
	def __init__(self, number_of_weights):
		np.random.seed(0)
		self.w = np.random.randn(number_of_weights,1)
		self.b = 0.

	def load_data(self):
		datafile = "housing.data"
		data = np.fromfile(datafile,sep=" ")
		
		feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',\
			'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
		feature_num = len(feature_names)
		data = data.reshape([data.shape[0] // feature_num,feature_num])
		ratio = 0.8
		offset = int(data.shape[0] * ratio)
		training_data = data[:offset]
		maximums,minimums,avgs = training_data.max(axis=0),training_data.min(axis=0),training_data.sum(axis=0) / training_data.shape[0]
		for i in range(feature_num):
			data[:,i] = (data[:,i] - minimums[i]) / (maximums[i] - minimums[i])
		training_data = data[:offset]
		test_data = data[offset:]
		return training_data,test_data

	def forword(self,x):
		z = np.dot(x,self.w) + self.b
		return z

	def loss(self,z,y):
		error = z - y
		num_samples = error.shape[0]
		cost = error * error
		cost = np.sum(cost) / num_samples
		return cost
	
	def gradient(self,x,y):
		z = self.forword(x)
		gradient_w = (z - y) * x
		gradient_w = np.mean(gradient_w,axis=0)
		gradient_b = z - y
		gradient_b = np.mean(gradient_b)
		return gradient_w,gradient_b

	def updata(self,gradient_w,gradient_b,eta=0.01):
		self.w = self.w - eta * gradient_w
		self.b = self.b - eta * gradient_b

	def train(self,training_data,num_epoches,batch_size=10,eta=0.01):
		n = len(training_data)
		losses = []
		for epoch_id in range(num_epoches):
			np.random.shuffle(training_data)
			mini_batches = [training_data[k:k + batch_size] for k in range(0,n,batch_size)]
			for iter_id,mini_batch in enumerate(mini_batches):
				x = mini_batch[:,:-1]
				y = mini_batch[:,-1:]
				z = self.forword(x)
				L = self.loss(z,y)
				gradient_w,gradient_b = self.gradient(x,y)
				self.updata(gradient_w,gradient_b,eta)
				losses.append(L)
				print('Epoch {:3d} / iter {:3d}, loss={:4f}'.format(epoch_id,iter_id,L))
		return losses


class Netswork(object):
	def __init__(self, number_of_weights,n):
		np.random.seed(0)
		self.w = np.random.randn(number_of_weights,n)
		self.b = np.random.randn(number_of_weights)

	def load_data(self):
		datafile = "housing.data"
		data = np.fromfile(datafile,sep=" ")
		
		feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',\
			'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
		feature_num = len(feature_names)
		data = data.reshape([data.shape[0] // feature_num,feature_num])
		ratio = 0.8
		offset = int(data.shape[0] * ratio)
		training_data = data[:offset]
		maximums,minimums,avgs = training_data.max(axis=0),training_data.min(axis=0),training_data.sum(axis=0) / training_data.shape[0]
		for i in range(feature_num):
			data[:,i] = (data[:,i] - minimums[i]) / (maximums[i] - minimums[i])
		training_data = data[:offset]
		test_data = data[offset:]
		return training_data,test_data

	def forword(self,x):
		z = np.dot(x,self.w) + self.b
		return z

	def loss(self,z,y):
		error = z - y
		num_samples = error.shape[0]
		cost = error * error
		cost = np.sum(cost) / num_samples
		return cost
	
	def gradient(self,x,y):
		z = self.forword(x)
		gradient_w = (z - y) * x
		gradient_w = np.mean(gradient_w,axis=0)
		gradient_b = z - y
		gradient_b = np.mean(gradient_b)
		return gradient_w,gradient_b

	def updata(self,gradient_w,gradient_b,eta=0.01):
		self.w = self.w - eta * gradient_w
		self.b = self.b - eta * gradient_b

	def train(self,training_data,num_epoches,batch_size=10,eta=0.01):
		n = len(training_data)
		losses = []
		for epoch_id in range(num_epoches):
			np.random.shuffle(training_data)
			mini_batches = [training_data[k:k + batch_size] for k in range(0,n,batch_size)]
			for iter_id,mini_batch in enumerate(mini_batches):
				x = mini_batch[:,:-1]
				y = mini_batch[:,-1:]
				z = self.forword(x)
				L = self.loss(z,y)
				gradient_w,gradient_b = self.gradient(x,y)
				self.updata(gradient_w,gradient_b,eta)
				losses.append(L)
				print('Epoch {:3d} / iter {:3d}, loss={:4f}'.format(epoch_id,iter_id,L))
		return losses

net = Network(13)
training_data,test_data = net.load_data()
#x = train_data[:,:-1]
#y = train_data[:,-1:]
num_iterations = 100
losses = net.train(training_data,50,100,eta=0.1)
plot_x=np.arange(len(losses))
plot_y =np.array(losses)
plt.plot(plot_x,plot_y)
plt.show()