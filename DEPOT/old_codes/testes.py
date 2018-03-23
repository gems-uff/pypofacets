# myPOFacets1Mono20170607.py
# myPOFacets1 v0.1mono
# Inspirado no programa POFacets noGUI v2.2 em MATLAB
import numpy as np
# import matplotlib as mp

name = "/home/clayton/Documentos/POFACETS/pofacets2.2nogui/BOX"
fname = name + "/coordinates.m"
print(fname)
coordinates = np.loadtxt(fname)
print(coordinates)
xpts = coordinates[:, 0]
print(xpts)
ypts = coordinates[:, 1]
print(ypts)
zpts = coordinates[:, 2]
print(zpts)
nverts = len(xpts)
print(nverts)
fname2 = name + "/facets.m"
print(fname2)
facets = np.loadtxt(fname2)
print(facets)
nfcv = facets[:, 0]
print(nfcv)
node1 = facets[:, 1]
print(node1)
node2 = facets[:, 2]
print(node2)
node3 = facets[:, 3]
print(node3)
iflag = 0
ilum = facets[:, 4]
Rs = facets[:, 5]
ntria = len(node3)
print(ntria)
vind = [[node1[i], node2[i], node3[i]]
        for i in range(ntria)]
print(vind)

