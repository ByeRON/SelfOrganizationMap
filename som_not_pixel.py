import sys, os
sys.path.append(os.pardir)

import numpy as np
from matplotlib import pyplot as plt
from dataset.load_vec1d import load_vec


class SOM():

	def __init__(self, teachers, N, seed=None):
		self.teachers = np.array(teachers)
		self.n_teacher = self.teachers.shape[0]
		self.N = N
		if not seed is None:
			np.random.seed(seed)
		
		x, y = np.meshgrid(range(self.N), range(self.N))
		"""
		self.c = np.hstack((x.flatten()[:, np.newaxis],
							y.flatten()[:, np.newaxis]))
		"""
		self.c = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
		"""
		self.nodes = np.random.rand(self.N*self.N,self.teachers.shape[1])
		self.nodes = np.random.randint(0,2,(self.N*self.N,self.teachers.shape[1]))
		"""
		self.nodes = np.random.rand(self.N*self.N,self.teachers.shape[1])

	def train(self):
		for i, teacher in enumerate(self.teachers):
			bmu = self._best_matching_unit(teacher)
			d = np.linalg.norm(self.c - bmu, axis=1)
			L = self._learning_ratio(i)
			S = self._learning_radius(i, d)
			"""
			self.nodes += L * S[:, np.newaxis] * (teacher - self.nodes)
			"""
			self.nodes += L * S.reshape(-1,1) * (teacher - self.nodes)
			print(i)
		return self.nodes

	def _best_matching_unit(self, teacher):
		norms = np.linalg.norm(self.nodes - teacher, axis=1)
		bmu = np.argmin(norms)
		return np.unravel_index(bmu,(self.N, self.N))

	def _neighbourhood(self, t):
		halflife = float(self.n_teacher/4) 
		initial  = float(self.N/2)
		return initial*np.exp(-t/halflife)

	def _learning_ratio(self, t):
		halflife = float(self.n_teacher/4)
		initial  = 0.1
		return initial*np.exp(-t/halflife)

	def _learning_radius(self, t, d):
		s = self._neighbourhood(t)
		return np.exp(-d**2/(2*s**2))

def main():
	N = 20
	(x_train, t_train), (x_test, t_test) = load_vec(normalize=True)
	teachers = np.array(x_test)
	print(x_train[0].reshape(300,200)[150,:])
	#plt.imshow(teachers[0].reshape(300,200), cmap='gray', interpolation='none')
	#plt.show()

	som = SOM(teachers, N=N, seed=50)

	# Initial map
	init_map = np.sum(som.nodes.reshape(N,N,200*300), axis=2)
	#print(init_map)
	#plt.imshow(init_map, cmap='gray', interpolation='none')
	#plt.show()
	#plt.imshow(som.nodes.reshape((N, N, 3)), interpolation='none')
	#plt.show()

	# Train
	som.train()

	# Trained MAP
	trained_map = np.sum(som.nodes.reshape(N,N,200*300), axis=2)
	#print(trained_map)
	#print(np.max(trained_map))
	trained_map = trained_map / np.max(trained_map)
	#print(trained_map)
	#trained_map = trained_map / 10000
	plt.imshow(trained_map, cmap='gray', interpolation='none')
	plt.show()

if __name__ == '__main__':
	main()
