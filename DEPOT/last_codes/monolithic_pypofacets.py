import sys
import math
import numpy as np

from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

input_model = sys.argv[1]
input_data_file = sys.argv[2]

params = open(input_data_file, 'r')
param_list = []
for line in params:
    if not line.startswith("#"):
        param_list.append(int(line))
freq, corr, delstd, ipol, pstart, pstop, delp, tstart, tstop, delt = param_list
params.close()

c = 3e8
waveL = c / freq

corel = corr / waveL
delsq = delstd ** 2
bk = 2 * math.pi / waveL
cfact1 = math.exp(-4 * bk ** 2 * delsq)
cfact2 = 4 * math.pi * (bk * corel) ** delsq
rad = math.pi / 180
Lt = 0.05
Nt = 5

if ipol == 0:
    Et = 1 + 0j
    Ep = 0 + 0j
elif ipol == 1:
    Et = 0 + 0j
    Ep = 1 + 0j
Co = 1

fname = input_model + "/coordinates.m"
coordinates = np.loadtxt(fname)
xpts = coordinates[:, 0]
ypts = coordinates[:, 1]
zpts = coordinates[:, 2]
nverts = len(xpts)

fname2 = input_model + "/facets.m"
facets = np.loadtxt(fname2)

nfcv = facets[:, 0]
node1 = facets[:, 1]
node2 = facets[:, 2]
node3 = facets[:, 3]
iflag = 0
ilum = facets[:, 4]
Rs = facets[:, 5]
ntria = len(node3)

vind = [[node1[i], node2[i], node3[i]]
        for i in range(ntria)]
x = xpts
y = ypts
z = zpts
r = [[x[i], y[i], z[i]]
     for i in range(nverts)]

fig1 = plt.figure()
ax = Axes3D(fig1)
for i in range(ntria):
    Xa = [int(r[int(vind[i][0])-1][0]), int(r[int(vind[i][1])-1][0]), int(r[int(vind[i][2])-1][0]), int(r[int(vind[i][0])-1][0])]
    Ya = [int(r[int(vind[i][0])-1][1]), int(r[int(vind[i][1])-1][1]), int(r[int(vind[i][2])-1][1]), int(r[int(vind[i][0])-1][1])]
    Za = [int(r[int(vind[i][0])-1][2]), int(r[int(vind[i][1])-1][2]), int(r[int(vind[i][2])-1][2]), int(r[int(vind[i][0])-1][2])]
    ax.plot3D(Xa, Ya, Za)
    ax.set_xlabel("X Axis")
ax.set_title("3D Model: " + input_model)
plt.savefig("teste.png")
plt.close()

if delp == 0:
    delp = 1
if pstart == pstop:
    phr0 = pstart*rad
if delt == 0:
    delt = 1
if tstart == tstop:
    thr0 = tstart*rad
it = math.floor((tstop-tstart)/delt)+1
ip = math.floor((pstop-pstart)/delp)+1

areai = []
beta = []
alpha = []
for i in range(ntria):
    A0 = ((r[int(vind[i][1])-1][0]) - (r[int(vind[i][0])-1][0]))
    A1 = ((r[int(vind[i][1])-1][1]) - (r[int(vind[i][0])-1][1]))
    A2 = ((r[int(vind[i][1])-1][2]) - (r[int(vind[i][0])-1][2]))
    A = [int(A0), int(A1), int(A2)]
    B0 = ((r[int(vind[i][2]) - 1][0]) - (r[int(vind[i][1]) - 1][0]))
    B1 = ((r[int(vind[i][2]) - 1][1]) - (r[int(vind[i][1]) - 1][1]))
    B2 = ((r[int(vind[i][2]) - 1][2]) - (r[int(vind[i][1]) - 1][2]))
    B = [int(B0), int(B1), int(B2)]
    C0 = ((r[int(vind[i][0]) - 1][0]) - (r[int(vind[i][2]) - 1][0]))
    C1 = ((r[int(vind[i][0]) - 1][1]) - (r[int(vind[i][2]) - 1][1]))
    C2 = ((r[int(vind[i][0]) - 1][2]) - (r[int(vind[i][2]) - 1][2]))
    C = [int(C0), int(C1), int(C2)]
    N = -(np.cross(B,A))
    d = [np.linalg.norm(A), np.linalg.norm(B), np.linalg.norm(C)]
    ss = 0.5*sum(d)
    areai.append(math.sqrt(ss*(ss-np.linalg.norm(A))*(ss-np.linalg.norm(B))*(ss-np.linalg.norm(C))))
    Nn = np.linalg.norm(N)
    N = N/Nn
    beta.append(math.acos(N[2]))
    alpha.append(math.atan2(N[1],N[0]))

phi = []
theta = []
D0 = []
R = []
e0 = []
now = datetime.now().strftime("%Y%m%d%H%M%S")
filename_R = "R_monolithic_pypofacets_" + now + ".dat"
filename_E0 = "E0_monolithic_pypofacets_" + now + ".dat"
fileR = open(filename_R, 'w')
fileE0 = open(filename_E0, 'w')
r_data = [
    now, sys.argv[0], sys.argv[1], sys.argv[2],
    freq, corr, delstd, ipol, pstart, pstop,
    delp, tstart, tstop, delt
]
text = '\n'.join(map(str, r_data)) + '\n'
#fileR.write(text)
#fileE0.write(text)
for i1 in range(0, int(ip)):
    for i2 in range(0, int(it)):
        phi.append(pstart+i1*delp)
        phr = phi[i2]*rad
        theta.append(tstart+i2*delt)
        thr = theta[i2]*rad
        st = math.sin(thr)
        ct = math.cos(thr)
        cp = math.cos(phr)
        sp = math.sin(phr)
        u = st*cp
        v = st*sp
        w = ct
        D0.append([u, v, w])
        U = u
        V = v
        W = w
        uu = ct*cp
        vv = ct*sp
        ww = -st
        fileR.write(str(i2))
        fileR.write(" ")
        fileR.write(str([u, v, w]))
        fileR.write("\n")
        R.append([u, v, w])
        fileE0.write(str(i2))
        fileE0.write(" ")
        fileE0.write(str([(uu*Et-sp*Ep), (vv*Et+cp*Ep), (ww*Et)]))
        fileE0.write("\n")
        e0.append([(uu*Et-sp*Ep), (vv*Et+cp*Ep), (ww*Et)])
fileR.close()
fileE0.close()
