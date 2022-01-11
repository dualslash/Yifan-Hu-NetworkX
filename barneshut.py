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

node_num = 1

class Quadtree:
	def __init__(self, min_x, min_y, size, max_dep):
		self.min_x = min_x
		self.min_y = min_y
		self.size = size
		self.isleaf = True
		self.max_dep = max_dep
		self.mass = 0
		self.children = []
		self.mass_center = [0.0, 0.0]

	def quad_divide(self):
		global axes
		if self.max_dep == 0:
			if len(self.children) == 0:
				self.children = [self]
		else:
			children_size = self.size / 2.0

			self.isleaf = False
			self.children.append(Quadtree(self.min_x, self.min_y + children_size, children_size, self.max_dep - 1))
			self.children.append(Quadtree(self.min_x + children_size, self.min_y + children_size, children_size, self.max_dep - 1))
			self.children.append(Quadtree(self.min_x, self.min_y, children_size, self.max_dep - 1))
			self.children.append(Quadtree(self.min_x + children_size, self.min_y, children_size, self.max_dep - 1))

	def insert_node(self, node):
		if self.min_x <= node[0] <= self.min_x + self.size and self.min_y <= node[1] <= self.min_y + self.size:

			ori_node = [i for i in self.mass_center]
			self.assimilateNode(node)

			if self.isleaf:
			
				self.quad_divide()
				ori_update = new_update = False
				
				for i in self.children:
					if ori_node[0] * ori_node[1] != 0.0 and i.min_x <= ori_node[0] <= i.min_x + i.size and i.min_y <= ori_node[1] <= i.min_y + i.size:

						i.quad_insert(ori_node)
						ori_update = True
						
						if i.min_x <= node[0] <= i.min_x + i.size and i.min_y <= node[1] <= i.min_y + i.size:
							if i.insert_node(node):
								return True
					if i.quad_insert(node):
						new_update = True
					if ori_update and new_update:
						return True
			else:
				for i in self.children:
					if i.insert_node(node):
						return True
		else:
			return False

	def quad_insert(self, node):
		global node_num
		if self.min_x <= node[0] <= self.min_x + self.size and self.min_y <= node[1] <= self.min_y + self.size:

			self.assimilateNode(node)
			node_num += 1

			return True
		else:
			return False

	def assimilateNode(self, node):
		self.mass_center[0] = (self.mass * self.mass_center[0] + node[0]) / (self.mass + 1)
		self.mass_center[1] = (self.mass * self.mass_center[1] + node[1]) / (self.mass + 1)
		self.mass += 1

	def get_farthest_mass(self, node, threshold, pos_arr):
		if node[0] == self.mass_center[0] and node[1] == self.mass_center[1]:
			return
		if self.size / math.sqrt((node[0] - self.mass_center[0]) ** 2 + (node[1] - self.mass_center[1]) ** 2) < threshold\
				and not self.isleaf:
			pos_arr.append([self.mass, self.mass_center])
			return
		elif self.isleaf and self.mass != 0:
			pos_arr.append([self.mass, self.mass_center])
		for i in self.children:
			i.get_farthest_mass(node, threshold, pos_arr)

	def get_all_mass_center(self, centers):
		if len(self.children) == 0:
			return
		for i in self.children:
			if i.mass_center[0] * i.mass_center[1] != 0.0:

				centers.append([i.size, i.mass, i.mass_center])
				i.get_all_mass_center(centers)

	def cal_electrical_forces(self, node, threshold, c, k, e_force_vector):

		if node[0] == self.mass_center[0] and node[1] == self.mass_center[1]:
			return
		if self.size / math.sqrt(
								(node[0] - self.mass_center[0]) ** 2 + (node[1] - self.mass_center[1]) ** 2) < threshold \
				and not self.isleaf:

			distance_sqr = (self.mass_center[0] - node[0]) ** 2 + (self.mass_center[1] - node[1]) ** 2
			e_force_vector[0] += (node[0] - self.mass_center[0]) * (c * k ** 2 / distance_sqr) * self.mass
			e_force_vector[1] += (node[1] - self.mass_center[1]) * (c * k ** 2 / distance_sqr) * self.mass

			return

		elif self.isleaf and self.mass != 0:

			distance_sqr = (self.mass_center[0] - node[0]) ** 2 + (self.mass_center[1] - node[1]) ** 2
			e_force_vector[0] += (node[0] - self.mass_center[0]) * (c * k ** 2 / distance_sqr)
			e_force_vector[1] += (node[1] - self.mass_center[1]) * (c * k ** 2 / distance_sqr)

		if self.children == [self]:
			return
		for i in self.children:
			i.cal_electrical_forces(node, threshold, c, k, e_force_vector)

if __name__ == "__main__":

	print("Using Barnes-Hut Quadtree Module")