from bitpack import BPDR
from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris()
iris_data = iris.data
iris_targets = iris.target

bpdr = BPDR(n_components=2)

bpdr_data = bpdr.fit_transform(iris_data, iris_targets)
print(bpdr.variances)
x1 = []
x2 = []
for row in bpdr_data:
    x1.append(row[0])
    x2.append(row[1])
    
plt.scatter(x1, x2, c=iris_targets, cmap=plt.cm.Set1, edgecolor='k')
plt.show()

