# myPOFacets1Mono20170607.py
# myPOFacets1 v0.1mono
# Inspirado no programa POFacets noGUI v2.2 em MATLAB
import math
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

print("PyPOFacets - v0.1")
print("=================")
# freq = float(input("Entre com a frequencia do radar em Hz:"))
freq = 15000000
print(freq)
print("Frequencia radar em Hz:", freq, "Hz")
c = 3e8
wave = c / freq
print(c)
print("Comprimento de onda em metros:", wave, "m")
# corr = float(input("Entre com a distancia de correlacao em metros:"))
corr = 0
corel = corr / wave
# delstd = float(input("Entre com o desvio padrao em metros:"))
delstd = 0
delsq = delstd ** 2
bk = 2 * math.pi / wave
cfact1 = math.exp(-4 * bk ** 2 * delsq)
cfact2 = 4 * math.pi * (bk * corel) ** delsq
rad = math.pi / 180
Lt = 0.05
Nt = 5
# ipol = float(input("Entre com a polarizacao de onda incidente:"))
ipol = 0
if ipol == 0:
    Et = 1 + 0j
    Ep = 0 + 0j
elif ipol == 1:
    Et = 0 + 0j
    Ep = 1 + 0j
else:
    print("erro")
Co = 1
# name = input("Entre com o nome do diretorio do modelo:")
# name = "/home/clayton/Documentos/POFACETS/pofacets2.2nogui/BOX"
# name = "/home/clayton/Documentos/POFACETS/testes/BOX"
name = "./BOX"
# name = "/home/clayton/Documentos/POFACETS/pofacets2.2nogui/PLATE"
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
# vind = np.matrix([[node1[i], node2[i], node3[i]]
#    for i in range(ntria)])
print(vind)
x = xpts
y = ypts
z = zpts
r = [[x[i], y[i], z[i]]
     for i in range(nverts)]
# r = np.matrix([[x[i], y[i], z[i]]
#    for i in range(nverts)])
print(r)
# inicio plot
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
    # ax.plot_wireframe(Xa, Ya, Za)
    ax.plot(Xa, Ya, Za)

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
    ax.set_xlabel("testex")
ax.set_title("teste")
plt.show()
plt.close()

# Oct 138 - Pattern Loop
# pstart = float(input("Entre com o valor de referencia do angulo phi inicial em graus:"))
pstart = 0
# pstop = float(input("Entre com o valor de referencia do angulo phi final em graus:"))
pstop = 0
# delp = float(input("Entre com o passo do angulo phi em graus:"))
delp = 0

if delp == 0:
    delp = 1
if pstart == pstop:
    phr0 = pstart*rad

# tstart = float(input("Entre com o valor de referencia do angulo theta inicial em graus:"))
tstart = 0
# tstop = float(input("Entre com o valor de referencia do angulo theta final em graus:"))
tstop = 360
# delt = float(input("Entre com o passo do angulo theta em graus:"))
delt = 2

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
            # Test to see if front face is illuminated: AFAZER
            # Local direction cosines
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
# print(D0)
