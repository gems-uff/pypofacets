# myPOFacets1MonoEst20180222MonolithicEng.py : myPOFacetsMono.py translated
# Inspired by the software POFacets noGUI v2.2 in MATLAB

import math
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



# @begin POFacets
# @in  name  @as Name
# @in  fname @as CoordinatesFile  @URI file:{name}/coordinates.m
# @in  fname2 @as FacetsFile  @URI file:{name}/facets.m
# @out plot @as Plot
# @out e0 @as E0
# @out T1 @as T1
# @out R @as R
print("PyPOFacets - v0.1")
print("=================")
# freq = float(input("Enter the radar frequency in Hz:"))
freq = 15000000
print(freq)

# @begin CalculateWaveLength
# :in:  freq :as: Frequency
# :out: waveL :as: WaveLength
print("The radar frequency in Hz:", freq, "Hz")
c = 3e8
wave = c / freq
print(c)
print("Wavelength in meters:", wave, "m")
# @end CalculateWaveLength

# corr = float(input("Enter the correlation distance in meters:"))
corr = 0
corel = corr / wave
# delstd = float(input("Enter the standard deviation in meters:"))
delstd = 0
delsq = delstd ** 2
bk = 2 * math.pi / wave
cfact1 = math.exp(-4 * bk ** 2 * delsq)
cfact2 = 4 * math.pi * (bk * corel) ** delsq
rad = math.pi / 180
Lt = 0.05
Nt = 5
# ipol = float(input("Enter the incident wave polarization:"))
ipol = 0

# @begin CalculateIncidentWavePolarization
# :in: ipol :as: InputPolarization
# @out Et
# @out Ep
if ipol == 0:
    Et = 1 + 0j
    Ep = 0 + 0j
elif ipol == 1:
    Et = 0 + 0j
    Ep = 1 + 0j
else:
    print("erro")
Co = 1
# @end CalculateIncidentWavePolarization

# name = input("Enter the directory name for data:")
# name = "/home/clayton/Documentos/POFACETS/pofacets2.2nogui/BOX"
# name = "/home/clayton/Documentos/POFACETS/pofacets2.2nogui/PLATE"
#name = "/home/clayton/PycharmProjects/PyPOFacets/BOX"
name = "BOX"

# @begin ReadModelCoordinates
# @in  name @as Name
# @in  fname @as CoordinatesFile  @URI file:{name}/coordinates.m
# @out xpts @as XPoints
# @out ypts @as YPoints
# @out zpts @as ZPoints
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
# @end ReadModelCoordinates

# @begin ReadFacetsModel
# @in  name @as Name
# @in  fname2 @as FacetsFile  @URI file:{name}/facets.m
# @out facets @as Facets
fname2 = name + "/facets.m"
print(fname2)
facets = np.loadtxt(fname2)
print(facets)
# @end ReadFacetsModel

# @begin GenerateTransposeMatrix
# @in  facets @as Facets
# @out node1 @as Node1
# @out node1 @as Node2
# @out node3 @as Node3
# @out ntria @as NTria
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
# @end GenerateTransposeMatrix

# vind = np.matrix([[node1[i], node2[i], node3[i]]
#    for i in range(ntria)])
print(vind)

# @begin GenerateCoordinatesPoints
# @in  xpts @as XPoints
# @in  ypts @as YPoints
# @in  zpts @as ZPoints
# @out r @as Points
x = xpts
y = ypts
z = zpts
r = [[x[i], y[i], z[i]]
     for i in range(nverts)]
# r = np.matrix([[x[i], y[i], z[i]]
#    for i in range(nverts)])
print(r)
# @end GenerateCoordinatesPoints


# @begin Plot3dGraphModel
# @in  node1 @as Node1
# @in  node1 @as Node2
# @in  node3 @as Node3
# @in  r @as Points
# @out plot @as Plot
# start plot
fig1 = plt.figure()
ax = Axes3D(fig1)
# print(x[int(vind[0][0])])
# print(vind[0][0])
# int(vind[0][0])
# print(isinstance(int(vind[0][0]), float))
print(x)
for i in range(ntria):
    # print(int(vind[i][0]))
    # Xa = [x[int(vind[0][0])]]# , x[int(vind[i][1])], x[int(vind[i][2])], x[int(vind[i][0])]]
    Xa = [int(r[int(vind[i][0])-1][0]), int(r[int(vind[i][1])-1][0]), int(r[int(vind[i][2])-1][0]), int(r[int(vind[i][0])-1][0])]
    print(Xa)
    Ya = [int(r[int(vind[i][0])-1][1]), int(r[int(vind[i][1])-1][1]), int(r[int(vind[i][2])-1][1]), int(r[int(vind[i][0])-1][1])]
    # print(Ya)
    Za = [int(r[int(vind[i][0])-1][2]), int(r[int(vind[i][1])-1][2]), int(r[int(vind[i][2])-1][2]), int(r[int(vind[i][0])-1][2])]
    # print(Za)
    # ax.plot3D(Xa, Ya, Za)
    # ax.plot(Xa, Ya, Za)
    ax.plot_wireframe(Xa, Ya, Za)

    # ax.plot_surface(Xa, Ya, Za)
    # print(Xa)
#   for i in range(ntria)])
#    Xa = [x[i], x[i], x[i], x[i]]
#    Ya = [y(vind(i, 1)), y(vind(i, 2)), y(vind(i, 3)), y(vind(i, 1))]
#    Za = [z(vind(i, 1)), z(vind(i, 2)), z(vind(i, 3)), z(vind(i, 1))]
# Xa = [0, 1, 0, 0]
# Ya = [0, 0, 1, 0]
# Za = [0, 1, 1, 0]
#    ax.plot3D(Xa, Ya, Za)
    ax.set_xlabel("testx")
ax.set_title("test")
plt.show()
plt.close()
# @end Plot3dGraphModel

# Oct 138 - Pattern Loop
# pstart = float(input("Enter the start phi angle in degrees:"))
pstart = 0
# pstop = float(input("Enter the stop phi angle in degrees:"))
pstop = 0
# delp = float(input("Enter the phi increment (step) in degrees:"))
delp = 0
# tstart = float(input("Enter the start theta angle in degrees:"))
tstart = 0
# tstop = float(input("Enter the stop theta angle in degrees:"))
tstop = 360
# delt = float(input("Enter the theta increment (step) in degrees:"))
delt = 2

# @begin CalculateRefsGeometryModel
# :in:  pstart :as: PStart
# :in:  pstop :as: PStop
# :in:  delp :as: InputDelP
# :in:  tstart :as: TStart
# :in:  tstop :as: TStop
# :in:  delt :as: InputDelT
# :in:  rad :as: Rad
# @out it @as IT
# @out ip @as IP
# @out delp @as DelP
# @out delt @as DelT
if delp == 0:
    delp = 1
if pstart == pstop:
    phr0 = pstart*rad


if delt == 0:
    delt = 1
if tstart == tstop:
    thr0 = tstart*rad

it = math.floor((tstop-tstart)/delt)+1
print(it)
ip = math.floor((pstop-pstart)/delp)+1
print(ip)
print("last step")
# @end CalculateRefsGeometryModel

# @begin CalculateEdgesAndNormalTriangles
# @in  node1 @as Node1
# @in  node1 @as Node2
# @in  node3 @as Node3
# @in  r @as Points
# :out: areai :as: AreaI
# @out beta @as Beta
# @out alpha @as Alpha
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

# @end CalculateEdgesAndNormalTriangles

phi = []
theta = []
D0 = []
R = []
e0 = []
# print("ok")
print(ip)
print(it)
for i1 in range(0, ip):
    # print("ok")
    for i2 in range(0, it):

        # @begin CalculateGlobalAnglesAndDirections
        # @in  ip @as IP
        # @in  it @as IT
        # :in:  pstart :as: PStart
        # @in  delp @as DelP
        # :in:  rad :as: Rad
        # :in:  tstart :as: TStart
        # @in  delt @as DelT
        # :in:  phi :as: Phi
        # :in:  theta :as: Theta
        # @out u @as U
        # @out v @as V
        # @out w @as W
        # @out uu @as UU
        # @out vv @as VV
        # @out ww @as WW
        # @out sp @as SP
        # @out cp @as CP
        # @out D0 @as D0
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
        # @end CalculateGlobalAnglesAndDirections

        # Spherical coordinate system radial unit vector

        # @begin CalculateSphericalCoordinateSystemRadialUnitVector
        # @in  u @as U
        # @in  v @as V
        # @in  w @as W
        # @out R @as R
        R.append([u, v, w])
        # @end CalculateSphericalCoordinateSystemRadialUnitVector

        # Incident field in global Cartesian coordinates

        # @begin CalculateIncidentFieldInGlobalCartesianCoordinates
        # @in  uu @as UU
        # @in  vv @as VV
        # @in  ww @as WW
        # @in  Et @as Et
        # @in  Ep @as Ep
        # @in  sp @as SP
        # @in  cp @as CP
        # @out e0 @as E0
        e0.append([(uu*Et-sp*Ep), (vv*Et+cp*Ep), (ww*Et)])
        # print(e0)
        # e0.append(vv*Et+cp*Ep)
        # e0.append(ww*Et)
        # @end CalculateIncidentFieldInGlobalCartesianCoordinates

        # Begin loop over triangles
        sumt = 0
        sump = 0
        sumdt = 0
        sumdp = 0
        for m in range(ntria):
            # OctT 236
            # Test to see if front face is illuminated: FUT
            # Local direction cosines

            # @begin CalculateIlumFaces
            # @in  ntria @as NTria
            # @in D0 @as D0
            # @in ip @as IP
            # @in alpha @as Alpha
            # @in beta @as Beta
            # @out T1 @as T1
            ca = math.cos(alpha[m])
            sa = math.sin(alpha[m])
            cb = math.cos(beta[m])
            sb = math.sin(beta[m])
            T1 = []
            T1 = [[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]]
            print(T1)
            T2 = []
            T2 = [[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]]
            # print(T2)
            Dzero = np.array(D0[i1])
            # print(Dzero)
            # print(Dzero.transpose())
            D1 = T1*Dzero.transpose()
            D2 = T2*D1
            # print([i1], [i2], D2)
            # @end CalculateIlumFaces


# print(D0)

# @end POFacets