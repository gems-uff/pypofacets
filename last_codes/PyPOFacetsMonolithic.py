# PyPOFacetsMonolithic - Monostatic version
# Inspired by the software POFacets noGUI v2.2 in MATLAB - http://faculty.nps.edu/jenn/

import sys
import math
import numpy as np

from datetime import datetime
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

start = timer()
now = datetime.now().strftime("%Y%m%d%H%M%S")
print now
print "PyPOFacets - v0.1"
print "================="
print "\nScript:", sys.argv[0]
# Read and print 3D model package
input_model = sys.argv[1]
print "\n3D Model:", input_model
# Read and print input data file: pattern -> input_data_file_xxx.dat
input_data_file = sys.argv[2]
print "\nInput data file:", input_data_file
# Open input data file and gather parameters
params = open(input_data_file, 'r')
# 1: radar frequency
params.readline()
freq = int(params.readline())
print "\nThe radar frequency in Hz:", freq, "Hz"
c = 3e8
wave = c / freq
print "\nWavelength in meters:", wave, "m"
# 2: correlation distance in meters
params.readline()
corr = int(params.readline())
print "\nCorrelation distance in meters:", corr, "m"
corel = corr / wave
# 3: standard deviation in meters
params.readline()
delstd = int(params.readline())
print "\nStandard deviation in meters:", delstd, "m"
delsq = delstd ** 2
bk = 2 * math.pi / wave
cfact1 = math.exp(-4 * bk ** 2 * delsq)
cfact2 = 4 * math.pi * (bk * corel) ** delsq
rad = math.pi / 180
Lt = 0.05
Nt = 5
# 4: incident wave polarization
params.readline()
ipol = int(params.readline())
print "\nIncident wave polarization:", ipol
if ipol == 0:
    Et = 1 + 0j
    Ep = 0 + 0j
elif ipol == 1:
    Et = 0 + 0j
    Ep = 1 + 0j
else:
    print("erro")
Co = 1

# Processing 3D model
print "\nProcessing 3D model..."
name = input_model
fname = name + "/coordinates.m"
print "\n...", fname
coordinates = np.loadtxt(fname)
print coordinates
xpts = coordinates[:, 0]
print xpts
ypts = coordinates[:, 1]
print ypts
zpts = coordinates[:, 2]
print zpts
nverts = len(xpts)
print nverts
fname2 = name + "/facets.m"
print "\n...", fname2
facets = np.loadtxt(fname2)
print facets
nfcv = facets[:, 0]
print nfcv
node1 = facets[:, 1]
print node1
node2 = facets[:, 2]
print node2
node3 = facets[:, 3]
print node3
iflag = 0
ilum = facets[:, 4]
Rs = facets[:, 5]
ntria = len(node3)
print ntria
vind = [[node1[i], node2[i], node3[i]]
        for i in range(ntria)]
x = xpts
y = ypts
z = zpts
r = [[x[i], y[i], z[i]]
     for i in range(nverts)]

# Start plot
# print "\n... start plot..."
# fig1 = plt.figure()
# ax = Axes3D(fig1)
# for i in range(ntria):
#     Xa = [int(r[int(vind[i][0])-1][0]), int(r[int(vind[i][1])-1][0]), int(r[int(vind[i][2])-1][0]), int(r[int(vind[i][0])-1][0])]
#     Ya = [int(r[int(vind[i][0])-1][1]), int(r[int(vind[i][1])-1][1]), int(r[int(vind[i][2])-1][1]), int(r[int(vind[i][0])-1][1])]
#     Za = [int(r[int(vind[i][0])-1][2]), int(r[int(vind[i][1])-1][2]), int(r[int(vind[i][2])-1][2]), int(r[int(vind[i][0])-1][2])]
#     ax.plot3D(Xa, Ya, Za)
#     # ax.plot(Xa, Ya, Za) # same above
#     # ax.plot_wireframe(Xa, Ya, Za) # one color
#     # ax.plot_surface(Xa, Ya, Za) # does not work
#     ax.set_xlabel("X Axis")
# ax.set_title("3D Model: " + input_model)
# plt.show()
# plt.close()

# Oct 138 - Pattern Loop
# 5: start phi angle in degrees
params.readline()
pstart = int(params.readline())
print "\nStart phi angle in degrees:", pstart

# 6: stop phi angle in degrees
params.readline()
pstop = int(params.readline())
print "\nStop phi angle in degrees:", pstop

# 7: phi increment (step) in degrees
params.readline()
delp = int(params.readline())
print "\nPhi increment (step) in degrees:", delp

if delp == 0:
    delp = 1
if pstart == pstop:
    phr0 = pstart*rad

# 8: start theta angle in degrees
params.readline()
tstart = int(params.readline())
print "\nStart theta angle in degrees:", tstart

# 9: stop theta angle in degrees
params.readline()
tstop = int(params.readline())
print "\nStop theta angle in degrees:", tstop

# 10: theta increment (step) in degrees
params.readline()
delt = int(params.readline())
print "\nTheta increment (step) in degrees:", delt

if delt == 0:
    delt = 1
if tstart == tstop:
    thr0 = tstart*rad

it = math.floor((tstop-tstart)/delt)+1
print(it)
ip = math.floor((pstop-pstart)/delp)+1
print(ip)
params.close()

print("last step")
areai = []
beta = []
alpha = []

# OctT 168 - Get edge vectors and normal from edge cross products
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
    print(N)

    # OctT 184 - Edge lengths for triangle "i"
    d = [np.linalg.norm(A), np.linalg.norm(B), np.linalg.norm(C)]
    ss = 0.5*sum(d)
    areai.append(math.sqrt(ss*(ss-np.linalg.norm(A))*(ss-np.linalg.norm(B))*(ss-np.linalg.norm(C))))
    Nn = np.linalg.norm(N)
    # unit normals
    N = N/Nn
    # 0 < beta < 180
    beta.append(math.acos(N[2]))
    # -180 < phi < 180
    alpha.append(math.atan2(N[1],N[0]))
phi = []
theta = []
D0 = []
R = []
e0 = []
filename_R = "R_PyPOFacetsMonolithic_"+sys.argv[1]+"_"+sys.argv[2]+"_"+now+".dat"
print filename_R
filename_E0 = "E0_PyPOFacetsMonolithic_"+sys.argv[1]+"_"+sys.argv[2]+"_"+now+".dat"
print filename_E0
fileR = open(filename_R, 'w')
fileE0 = open(filename_E0, 'w')
fileR.write(now)
fileR.write("\n")
fileR.write(sys.argv[0])
fileR.write("\n")
fileR.write(sys.argv[1])
fileR.write("\n")
fileR.write(sys.argv[2])
fileR.write("\n")
fileR.write(str(freq))
fileR.write("\n")
fileR.write(str(corr))
fileR.write("\n")
fileR.write(str(delstd))
fileR.write("\n")
fileR.write(str(ipol))
fileR.write("\n")
fileR.write(str(pstart))
fileR.write("\n")
fileR.write(str(pstop))
fileR.write("\n")
fileR.write(str(delp))
fileR.write("\n")
fileR.write(str(tstart))
fileR.write("\n")
fileR.write(str(tstop))
fileR.write("\n")
fileR.write(str(delt))
fileR.write("\n")
fileE0.write(now)
fileE0.write("\n")
fileE0.write(sys.argv[0])
fileE0.write("\n")
fileE0.write(sys.argv[1])
fileE0.write("\n")
fileE0.write(sys.argv[2])
fileE0.write("\n")
fileE0.write(str(freq))
fileE0.write("\n")
fileE0.write(str(corr))
fileE0.write("\n")
fileE0.write(str(delstd))
fileE0.write("\n")
fileE0.write(str(ipol))
fileE0.write("\n")
fileE0.write(str(pstart))
fileE0.write("\n")
fileE0.write(str(pstop))
fileE0.write("\n")
fileE0.write(str(delp))
fileE0.write("\n")
fileE0.write(str(tstart))
fileE0.write("\n")
fileE0.write(str(tstop))
fileE0.write("\n")
fileE0.write(str(delt))
fileE0.write("\n")
for i1 in range(0, int(ip)):
    for i2 in range(0, int(it)):
        phi.append(pstart+i1*delp)
        phr = phi[i2]*rad
        # Global angles and direction cosines
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
        # Spherical coordinate system radial unit vector
        fileR.write(str(i2))
        fileR.write(" ")
        fileR.write(str([u, v, w]))
        fileR.write("\n")
        R.append([u, v, w])
        # Incident field in global Cartesian coordinates
        fileE0.write(str(i2))
        fileE0.write(" ")
        fileE0.write(str([(uu*Et-sp*Ep), (vv*Et+cp*Ep), (ww*Et)]))
        fileE0.write("\n")
        e0.append([(uu*Et-sp*Ep), (vv*Et+cp*Ep), (ww*Et)])
        # Begin loop over triangles
        sumt = 0
        sump = 0
        sumdt = 0
        sumdp = 0
        # for m in range(ntria):
            # OctT 236
            # Test to see if front face is illuminated: FUT
            # Local direction cosines
            # ca = math.cos(alpha[m])
            # sa = math.sin(alpha[m])
            # cb = math.cos(beta[m])
            # sb = math.sin(beta[m])
            # T1 = []
            # T1 = [[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]]
            # T2 = []
            # T2 = [[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]]
            # Dzero = np.array(D0[i1])
            # D1 = T1*Dzero.transpose()
            # D2 = T2*D1
fileR.close()
fileE0.close()
end = timer()
print end - start, "seg"
