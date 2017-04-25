from numpy import *
from ID3DTree import *
dtree = ID3DTree()
dtree.loadDataSet("dataset.dat", ["age","revenue","students","credit"])
dtree.train()
print dtree.tree