# Data Processing
import numpy as np
import pandas as pd

# Graph Processing
import networkx as nx

# Data Visualization
import matplotlib.pyplot as plt

# Misc Packages
import random
import math
import datetime

# Quadtree
from barneshut import Quadtree

class YifanHu:
	def __init__(self, graph, points):
		assert graph is not None
		assert points.shape[0] == graph.number_of_nodes() and points.shape[1] == 2
		self.pos = points
		self.G = graph
		self.node_num = points.shape[0]
		self.threshold = False
		self.delta = None
		self.A = None
		self.distance = None
		self.AverageEdgeLength = None
		self.relativeStrength = None
		self.stepRatio = None
		self.quadTreeMaxLevel = None
		self.barnesHutTheta = None
		self.convergenceThreshold = None
		self.adaptiveCooling = None
		self.optimalDistance = None
		self.step = None
		self.initialStep = None
		self.progress = None
		self.energy0 = None
		self.energy = None

		self.initPropertiesValues()
		self.min_energy = math.inf

	def getAverageEdgeLength(self):
		s1 = 3.2
		s2 = -0.8
		G = self.G.to_undirected()
		t = nx.to_numpy_matrix(self.G, weight='weight')
		self.A = nx.to_numpy_matrix(G, weight='weight')

		assert t.all() == self.A.all()
		self.A = np.asarray(self.A)
		self.delta = np.zeros((self.node_num, self.node_num, self.pos.shape[1]), dtype=self.A.dtype)
	
		for i in range(self.pos.shape[1]):
			self.delta[:, :, i] = self.pos[:, i, None] - self.pos[:, i]
		self.distance = np.sqrt((self.delta ** 2).sum(axis=-1))

		assert type(self.A) == type(self.distance)
		distance_x_2 = self.distance * self.A
		edge_num = len(np.nonzero(distance_x_2)[0]) / 2
		self.AverageEdgeLength = np.sum(distance_x_2) / 2 / edge_num

		for u, v in self.G.edges():
			self.A[u, v] = s1
			self.A[v, u] = s2

		return self.AverageEdgeLength

	def initPropertiesValues(self):
		self.progress = 0
		self.energy = 2 ** 31 - 1
		self.stepRatio = 0.95
		self.relativeStrength = 0.2
		self.quadTreeMaxLevel = 20
		self.barnesHutTheta = 1.2
		self.adaptiveCooling = True
		self.convergenceThreshold = 1e-4
		self.optimalDistance = self.relativeStrength ** (1.0 / 3) * self.getAverageEdgeLength()
		self.initialStep = self.optimalDistance / 5.0
		self.step = self.initialStep

	def resetPropertiesValues(self, stepRatio=0.95, relativeStrength=0.2, quadTreeMaxLevel=20,
	barnesHutTheta=1.2, convergenceThreshold=0.01): #1e-4
		assert 0.0 < stepRatio <= 1.0
		assert 0.0 < relativeStrength
		assert quadTreeMaxLevel > 10
		self.stepRatio = stepRatio
		self.relativeStrength = relativeStrength
		self.quadTreeMaxLevel = quadTreeMaxLevel
		self.barnesHutTheta = barnesHutTheta
		self.convergenceThreshold = convergenceThreshold
		self.optimalDistance = self.relativeStrength ** (1.0 / 3) * self.getAverageEdgeLength()
		self.initialStep = self.optimalDistance / 5.0
		self.step = self.initialStep

	def updateStep(self):
		if self.adaptiveCooling:
			if self.energy < self.energy0:
				self.progress += 1
				if self.progress >= 5:
					self.progress = 0
					self.step /= self.stepRatio
			else:
				self.progress = 0
		else:
			self.step /= self.stepRatio

	def control_run(self):
		self.updateStep()
		if self.min_energy > self.energy:
			print("Change ratio: %f" % (abs(self.energy - self.energy0) / self.energy))
		if abs(self.energy - self.energy0) / self.energy < self.convergenceThreshold:
			self.threshold = True

	def update(self):
		min_x, min_y = min(self.pos.T[0]), min(self.pos.T[1])
		size = max(max(self.pos.T[0]) - min(self.pos.T[0]), max(self.pos.T[1]) - min(self.pos.T[1]))
		BH_tree = Quadtree(min_x=min_x, min_y=min_y, size=size, max_dep=self.quadTreeMaxLevel)
	
		for i in self.pos:
			BH_tree.insert_node(i.tolist())
		e_forces_move = []

		for i in self.pos:
			e_force_vector = [0.0, 0.0]
			BH_tree.cal_electrical_forces(node=i.tolist(), threshold=self.barnesHutTheta,
			c=self.relativeStrength, k=self.optimalDistance, e_force_vector=e_force_vector)
			e_forces_move.append(e_force_vector)

		electric_forces_move = np.array(e_forces_move)
		s1 = 1
		s2 = 1
		spring_forces_move = np.zeros(shape=(self.node_num, 2))
	
		for u, v in self.G.edges():
			spring_forces_move[u, :] += s1 * (self.pos[u, :] - self.pos[v, :]) * self.distance[u, v] / self.optimalDistance
			spring_forces_move[v, :] -= s2 * (self.pos[u, :] - self.pos[v, :]) * self.distance[u, v] / self.optimalDistance

		assert electric_forces_move.shape == spring_forces_move.shape

		displacement = electric_forces_move - spring_forces_move
		self.energy0 = self.energy
		max_force, self.energy = get_force_norm(displacement)

		assert max_force > 0
		self.pos += displacement * (self.step / max_force)
		self.control_run()
	
		if self.energy < self.energy0:
			if self.energy < self.min_energy:
				self.min_energy = self.energy
				print("Energy: %d" % self.min_energy)
				plt.clf()
				self.G = self.G.to_undirected()
				nx.draw(self.G, self.pos, with_labels=False, node_size=0.5, width=0.1)
				if self.threshold:
					plt.pause(100) #100
				plt.pause(0.1)

def get_force_norm(forces):
	forces = np.sqrt((forces ** 2).sum(axis=1))
	forces = np.where(forces < 0.01, 0.01, forces)
 
	return max(1.0, np.max(forces)), np.sum(forces)

if __name__ == "__main__":

  print("Yifan Hu Algorithm Class")