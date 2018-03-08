# PyPOFacetsMonolithic - Monostatic version
# Inspired by the software POFacets noGUI v2.2 in MATLAB - http://faculty.nps.edu/jenn/

import sys
import math
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
# vind = np.matrix([[node1[i], node2[i], node3[i]]
#    for i in range(ntria)])
# print vind
x = xpts
y = ypts
z = zpts
r = [[x[i], y[i], z[i]]
     for i in range(nverts)]
# r = np.matrix([[x[i], y[i], z[i]]
#    for i in range(nverts)])
# print r
# start plot
print "\n... start plot..."
fig1 = plt.figure()
ax = Axes3D(fig1)
# print(x[int(vind[0][0])])
# print(vind[0][0])
# int(vind[0][0])
# print(isinstance(int(vind[0][0]), float))
for i in range(ntria):
    # print(int(vind[i][0]))
    # Xa = [x[int(vind[0][0])]]# , x[int(vind[i][1])], x[int(vind[i][2])], x[int(vind[i][0])]]
    Xa = [int(r[int(vind[i][0])-1][0]), int(r[int(vind[i][1])-1][0]), int(r[int(vind[i][2])-1][0]), int(r[int(vind[i][0])-1][0])]
    print(Xa)
    Ya = [int(r[int(vind[i][0])-1][1]), int(r[int(vind[i][1])-1][1]), int(r[int(vind[i][2])-1][1]), int(r[int(vind[i][0])-1][1])]
    # print(Ya)
    Za = [int(r[int(vind[i][0])-1][2]), int(r[int(vind[i][1])-1][2]), int(r[int(vind[i][2])-1][2]), int(r[int(vind[i][0])-1][2])]
    # print(Za)
    ax.plot3D(Xa, Ya, Za)
    # ax.plot(Xa, Ya, Za) # same above
    # ax.plot_wireframe(Xa, Ya, Za) # one color

    # ax.plot_surface(Xa, Ya, Za) # does not work
    # print(Xa)
    #   for i in range(ntria)])
    #    Xa = [x[i], x[i], x[i], x[i]]
    #    Ya = [y(vind(i, 1)), y(vind(i, 2)), y(vind(i, 3)), y(vind(i, 1))]
    #    Za = [z(vind(i, 1)), z(vind(i, 2)), z(vind(i, 3)), z(vind(i, 1))]
    # Xa = [0, 1, 0, 0]
    # Ya = [0, 0, 1, 0]
    # Za = [0, 1, 1, 0]
    #    ax.plot3D(Xa, Ya, Za)
    ax.set_xlabel("X Axis")
ax.set_title("3D Model: " + input_model)
plt.show()
plt.close()

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
print "\nStop theta angle in degrees:", tstart

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
print("last step")
# areai = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# print(areai)
areai = []
beta = []
alpha = []
# print(r[0][0])
# Get edge vectors and normal from edge cross products - OctT 168
for i in range(ntria):
    # X = [x(vind(i, 1)) x(vind(i, 2)) x(vind(i, 3)) x(vind(i, 1))];
    # Y = [y(vind(i, 1)) y(vind(i, 2)) y(vind(i, 3)) y(vind(i, 1))];
    # Z = [z(vind(i, 1)) z(vind(i, 2)) z(vind(i, 3)) z(vind(i, 1))];
    # Xa = [int(r[int(vind[i][0])-1][0]), int(r[int(vind[i][1])-1][0]), int(r[int(vind[i][2])-1][0]), int(r[int(vind[i][0])-1][0])]
    # Ya = [int(r[int(vind[i][0])-1][1]), int(r[int(vind[i][1])-1][1]), int(r[int(vind[i][2])-1][1]), int(r[int(vind[i][0])-1][1])]
    # Za = [int(r[int(vind[i][0])-1][2]), int(r[int(vind[i][1])-1][2]), int(r[int(vind[i][2])-1][2]), int(r[int(vind[i][0])-1][2])]
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
    # - r[int(vind[i][0])-1]
    # print(A0)
    # print(A1)
    # print(C)
    # print(B)
    # print(A)
    N = -(np.cross(B,A))
    print(N)
    # print(int(vind[i][1]))

    # Edge lengths for triangle "i" OctT 184
    d = [np.linalg.norm(A), np.linalg.norm(B), np.linalg.norm(C)]
    ss = 0.5*sum(d)
    # print(ss)
    areai.append(math.sqrt(ss*(ss-np.linalg.norm(A))*(ss-np.linalg.norm(B))*(ss-np.linalg.norm(C))))
    # print(areai)
    Nn = np.linalg.norm(N)
    # print(Nn)
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
# print("ok")
print(ip)
print(it)
for i1 in range(0, int(ip)):
    # print("ok")
    for i2 in range(0, int(it)):
        # print(i1)
        # print(i2)
        # phi.append(1)
        # phi.append(2)
        # print(phi)
        # print(pstart, i1, delp)
        phi.append(pstart+i1*delp)
        # print(phi)
        # print(phi[i2])
        phr = phi[i2]*rad
        # print(phr)
        # print(phi[i1][i2])
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
        # print(w)
        D0.append([u, v, w])
        # print(D0)
        # D0 = [u v w]
        U = u
        V = v
        W = w
        uu = ct*cp
        vv = ct*sp
        ww = -st
        # Spherical coordinate system radial unit vector
        R.append([u, v, w])
        # Incident field in global Cartesian coordinates
        e0.append([(uu*Et-sp*Ep), (vv*Et+cp*Ep), (ww*Et)])
        # print(e0)
        # e0.append(vv*Et+cp*Ep)
        # e0.append(ww*Et)
        # Begin loop over triangles
        sumt = 0
        sump = 0
        sumdt = 0
        sumdp = 0
        for m in range(ntria):
            # OctT 236
            # Test to see if front face is illuminated: FUT
            # Local direction cosines
            ca = math.cos(alpha[m])
            sa = math.sin(alpha[m])
            cb = math.cos(beta[m])
            sb = math.sin(beta[m])
            T1 = []
            T1 = [[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]]
            # print(T1)
            T2 = []
            T2 = [[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]]
            # print(T2)
            Dzero = np.array(D0[i1])
            # print(Dzero)
            # print(Dzero.transpose())
            D1 = T1*Dzero.transpose()
            D2 = T2*D1
            # print(D2)
            # print([i1], [i2], D2)
            # print(D0)