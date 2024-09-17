import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
from sklearn.cluster import KMeans


class K_Means:
	colors = ("red", "blue", "green", "gold", "silver", "purple", "pink", "orange", "black")
	def __init__(self, x, y, k):
		if len(x) != len(y):
			raise ValueError("Длины векторов не совпадают!")
		self.x = x
		self.y = y 
		self.k = k
		self.groups = [[]]
		i = np.random.randint(0, len(x)-1)
		self.centroids = [(x[i], y[i])]
		for _ in range(k-1):
			self.groups.append([])
			dist = []
			for i in range(len(x)):
				dist.append(min(self.distance(c[0], c[1], x[i], y[i]) for c in self.centroids))
			indx = dist.index(max(dist))
			self.centroids.append((x[indx], y[indx]))

	def distance(self, x1, y1, x2, y2):
		return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

	def get_groups(self):
		f = True
		while f:
			for k in range(self.k):
				group = []
				for i in range(len(self.x)):
					d = self.distance(self.centroids[k][0], self.centroids[k][1], self.x[i], self.y[i])
					if all(d < self.distance(self.centroids[j][0], self.centroids[j][1], self.x[i], self.y[i]) for j in range(self.k) if j != k):
						group.append((self.x[i], self.y[i]))
				self.groups[k] = group
				if len(group):
					new_centroid = tuple(np.mean(group, axis = 0))
					if new_centroid != self.centroids[k]:
						self.centroids[k] = new_centroid
						f = True
					else:
						f = False

		return self.groups

	def show(self):
		coords = self.get_groups()
		colors = self.colors
		for i in range(len(coords)):
			x = []
			y = []
			for j in range(len(coords[i])):
				x.append(coords[i][j][0])
				y.append(coords[i][j][1])
			plt.scatter(x, y, color = colors[i % len(colors)])
		plt.show()


class OptimalSchedule:
	def __init__(self, x, y):
		self.x = x 
		self.y = y 
 
	def show(self):
		square_sum = []
		for k in range(1, 20):
			obj = K_Means(x, y, k)
			groups = obj.get_groups()
			centroids = obj.centroids
			s = 0
			for i in range(len(groups)):
				group = np.array(groups[i])
				centroid = np.array(centroids[i])
				s += np.sum(np.sum((group - centroid)**2, axis = 1))
			square_sum.append(s)
		plt.plot(np.arange(1, 20), square_sum)
		plt.xticks(np.arange(1, 20, 1.0))
		plt.show()



x = np.array([2, 3, 7, 10, 5, 3, 4, 60, 70, 65, 75, 89, 69, 70, 30, 35, 50, 70, 65, 55, 39])
y = np.array([3, 5, 10, 2, 4, 8, 4, 3, 10, 9, 6, 4, 1, 2, 80, 57, 77, 65, 61, 59, 90])

iris_dataset = load_iris()
#x = iris_dataset["data"][:, 0]
#y = iris_dataset["data"][:, 3]

os = OptimalSchedule(x, y)
os.show()

k = 3

kmeans = K_Means(x, y, k)
kmeans.show()

data = list(zip(x, y))
kmeans = KMeans(n_clusters=k)
kmeans.fit(data)

print(kmeans.labels_)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()