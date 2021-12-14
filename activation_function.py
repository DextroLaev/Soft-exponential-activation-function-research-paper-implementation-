import numpy as np
import matplotlib.pyplot as plt

class soft_exponential:
	def __init__(self,alpha,x):
		self.alpha = alpha
		self.x = x

	def calculate(self,alpha):
		if alpha < 0:
			return -np.log(1-alpha*(self.x+alpha))/alpha
		elif alpha == 0:
			return self.x
		else:
			return ((np.exp(alpha*self.x)-1)/alpha) + alpha

	def function(self):
		result = []
		for i in self.alpha:
			sof_ex = self.calculate(i)
			result.append(sof_ex)
		self.plot_function(result)	
		return np.array(result)	

	def plot_function(self,output):
		for i in range(len(output)):
			plt.plot(self.x,output[i],label='{}'.format(self.alpha[i]))
		plt.xlim(-5,5)
		plt.ylim(-5,5)
		plt.xlabel('x')
		plt.ylabel("f(alpha,x)")
		plt.show()

if __name__ == '__main__':
	x = np.arange(-6,6,0.2)
	alpha = np.arange(-3,3,0.2)
	activation = soft_exponential(alpha,x)
	output = activation.function()
