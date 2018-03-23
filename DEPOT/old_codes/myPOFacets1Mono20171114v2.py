# myPOFacets1Mono20171114v2.py
# myPOFacets1 v0.1mono
# Inspirado no programa POFacets noGUI v2.2 em MATLAB
import math
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def calcula_comprimento_de_onda(freq):
    c = 3e8
    print(c)
    waveL = c / freq
    return waveL


def polarizacao_onda_incidente(ipol):
    if ipol == 0:
        Et = 1 + 0j
        Ep = 0 + 0j
    elif ipol == 1:
        Et = 0 + 0j
        Ep = 1 + 0j
    else:
        print("erro")
    return (Et, Ep)


def ler_coordenadas_modelo(name):
    fname = name + "/coordinates.m"
    print(fname)
    coordinates = np.around(np.loadtxt(fname)).astype(int)
    print(coordinates)
    return np.transpose(coordinates)


def ler_facets_modelo(name):
    fname2 = name + "/facets.m"
    print(fname2)
    facets = np.around(np.loadtxt(fname2)).astype(int)  #astype ver np.around(np.loadtxt(...))
    return facets





def matriz_transposta(facets):
    nfcv, node1, node2, node3, ilum, Rs = np.transpose(facets)
    return nfcv, node1, node2, node3, ilum, Rs


def main(name):
    print("PyPOFacets - v0.1")
    print("=================")
    # freq = float(input("Entre com a frequencia do radar em Hz:"))
    freq = 15000000
    print(freq)
    print("Frequencia radar em Hz:", freq, "Hz")

    waveL = calcula_comprimento_de_onda(freq)
    print("Comprimento de onda em metros:", waveL, "m")
    # corr = float(input("Entre com a distancia de correlacao em metros:"))
    # corr = 0
    # corel = corr / wave
    # delstd = float(input("Entre com o desvio padrao em metros:"))
    # delstd = 0
    # delsq = delstd ** 2
    # bk = 2 * math.pi / wave
    # cfact1 = math.exp(-4 * bk ** 2 * delsq)
    # cfact2 = 4 * math.pi * (bk * corel) ** delsq
    rad = math.pi / 180
    # Lt = 0.05
    # Nt = 5
    # ipol = float(input("Entre com a polarizacao de onda incidente:"))
    ipol = 1
    Et, Ep = polarizacao_onda_incidente(ipol)
    # print(Et)
    # print(Ep)
    # if ipol == 0:
    #     Et = 1 + 0j
    #     Ep = 0 + 0j
    # elif ipol == 1:
    #     Et = 0 + 0j
    #     Ep = 1 + 0j
    # else:
    #     print("erro")

    # Co = 1
    #
    # fname = name + "/coordinates.m"
    # print(fname)
    # coordinates = np.around(np.loadtxt(fname)).astype(int)
    # print(coordinates)
    xpts, ypts, zpts = ler_coordenadas_modelo(name)
    print(xpts)
    print(ypts)
    print(zpts)
    nverts = len(xpts)
    print(nverts)
    # fname2 = name + "/facets.m"
    # print(fname2)
    # facets = np.around(np.loadtxt(fname2)).astype(int)  #astype ver np.around(np.loadtxt(...))
    facets = ler_facets_modelo(name)
    print(facets)
    nfcv, node1, node2, node3, ilum, Rs = matriz_transposta(facets)
    print(nfcv)
    print(node1)
    print(node2)
    print(node3)
    ntria = len(node3)
    print(ntria)
    r = list(zip(xpts, ypts, zpts))
    print(r)
    # inicio plot
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    print(xpts)
    for point in zip(node1, node2, node3):
        Xa = [r[(point[0])-1][0], r[(point[1])-1][0], r[(point[2])-1][0], r[(point[0])-1][0]]
        print(Xa)
        Ya = [r[point[0] - 1][1], r[point[1] - 1][1], r[point[2] - 1][1], r[point[0] - 1][1]]
        Za = [r[point[0]-1][2], r[point[1]-1][2], r[point[2]-1][2], r[point[0]-1][2]]
        ax.plot_wireframe(Xa, Ya, Za)
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
    for point in zip(node1, node2, node3):
        A0 = ((r[point[1] - 1][0]) - (r[point[0] - 1][0]))
        A1 = ((r[point[1] - 1][1]) - (r[point[0] - 1][1]))
        A2 = ((r[point[1] - 1][2]) - (r[point[0] - 1][2]))
        A = [A0, A1, A2]
        B0 = ((r[point[2] - 1][0]) - (r[point[1] - 1][0]))
        B1 = ((r[point[2] - 1][1]) - (r[point[1] - 1][1]))
        B2 = ((r[point[2] - 1][2]) - (r[point[1] - 1][2]))
        B = [B0, B1, B2]
        C0 = ((r[point[0] - 1][0]) - (r[point[2] - 1][0]))
        C1 = ((r[point[0] - 1][1]) - (r[point[2] - 1][1]))
        C2 = ((r[point[0] - 1][2]) - (r[point[2] - 1][2]))
        C = [C0, C1, C2]
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
    # for i1 in range(0, ip):
    for i1 in range(ip):
        # print("ok")
        for i2 in range(it):
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


if __name__ == '__main__':

    # name = input("Entre com o nome do diretorio do modelo:")
    # name = "/home/clayton/Documentos/POFACETS/pofacets2.2nogui/PLATE"
    # name = "/home/clayton/Documentos/POFACETS/pofacets2.2nogu/BOX"
    name = "./BOX"
    main(name)
