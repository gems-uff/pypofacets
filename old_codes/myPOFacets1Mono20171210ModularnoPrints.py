# myPOFacets1Mono20171114v2.py
# myPOFacets1 v0.1mono
# Inspirado no programa POFacets noGUI v2.2 em MATLAB
import math
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from timeit import default_timer as timer


# ***funcao 1***
def calcular_comprimento_de_onda(freq):
    c = 3e8
    # print("velocidade da luz: ", c)
    waveL = c / freq
    return waveL


# ***funcao 2***
def calcular_polarizacao_onda_incidente(ipol):
    if ipol == 0:
        Et = 1 + 0j
        Ep = 0 + 0j
    elif ipol == 1:
        Et = 0 + 0j
        Ep = 1 + 0j
    else:
        print("erro")
    return (Et, Ep)


# ***funcao 3***
def ler_coordenadas_modelo(name):
    fname = name + "/coordinates.m"
    # print("Caminho e arquivo coordenadas modelo: ", fname)
    coordinates = np.around(np.loadtxt(fname)).astype(int)
    # print("Coordenadas dos pontos do modelo: ", coordinates)
    return np.transpose(coordinates)


# ***funcao 4***
def ler_facets_modelo(name):
    fname2 = name + "/facets.m"
    # print("Caminho e arquivo faces modelo: ", fname2)
    facets = np.around(np.loadtxt(fname2)).astype(int)  #astype ver np.around(np.loadtxt(...))
    return facets


# ***funcao 5***
def gerar_matriz_transposta(facets):
    nfcv, node1, node2, node3, ilum, Rs = np.transpose(facets)
    return nfcv, node1, node2, node3, ilum, Rs


# ***funcao 6***
def gerar_coordenadas_pontos(xpts, ypts, zpts):
    r = list(zip(xpts, ypts, zpts))
    return r


# ***funcao 7***
def plotar_grafico_3d_modelo(node1, node2, node3, r):
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    for point in zip(node1, node2, node3):
        Xa = [r[(point[0]) - 1][0], r[(point[1]) - 1][0], r[(point[2]) - 1][0], r[(point[0]) - 1][0]]
        # print(Xa)
        Ya = [r[point[0] - 1][1], r[point[1] - 1][1], r[point[2] - 1][1], r[point[0] - 1][1]]
        Za = [r[point[0] - 1][2], r[point[1] - 1][2], r[point[2] - 1][2], r[point[0] - 1][2]]
        # ax.plot_wireframe(Xa, Ya, Za)
        ax.plot(Xa, Ya, Za)
        ax.set_xlabel("testex")
    ax.set_title("teste")
    plt.show()
    plt.close()


# ***funcao 8***
def calcular_refs_geometria_modelo(pstart, pstop, delp, tstart, tstop, delt, rad):
    if delp == 0:
        delp = 1
    if pstart == pstop:
        phr0 = pstart*rad

    if delt == 0:
        delt = 1
    if tstart == tstop:
        thr0 = tstart*rad

    it = math.floor((tstop-tstart)/delt)+1
    # print("Quantidade de rotacoes horizontais na simulacao: ", it)
    ip = math.floor((pstop-pstart)/delp)+1
    # print("Quantidade de rotacoes verticais na simulacao: ", ip)

    return it, ip, delp, delt


# ***funcao 9***
def calcular_arestas_e_normal_triangulos(node1, node2, node3, r):
    areai = []
    beta = []
    alpha = []
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

        N = -(np.cross(B,A))
        # print("Refs. normal (bidim.): ", point, N)

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

    return areai, beta, alpha


# ***funcao 10***
def calcular_angulos_globais_e_direcoes(i1, i2, ip, it, pstart, delp, rad, tstart, delt, phi, theta, D0):
    phi.append(pstart + i1 * delp)
    phr = phi[i2] * rad
    # Global angles and direction cosines
    theta.append(tstart + i2 * delt)
    thr = theta[i2] * rad
    st = math.sin(thr)
    ct = math.cos(thr)
    cp = math.cos(phr)
    sp = math.sin(phr)
    u = st * cp
    v = st * sp
    w = ct
    D0.append([u, v, w])
    # D0 = [u v w]
    U = u
    V = v
    W = w
    uu = ct * cp
    vv = ct * sp
    ww = -st
    return u, v, w, uu, vv, ww, sp, cp, D0


# ***funcao 11***
def calcular_vetor_unitario_radial_sistema_coord_esfericas(u, v, w, R):
    R.append([u, v, w])
    return R


# ***funcao 12***
def calcular_campo_eletrico_incidente_coord_cartesianas_globais(uu, vv, ww, Et, Ep, sp, cp, e0):
    e0.append([(uu * Et - sp * Ep), (vv * Et + cp * Ep), (ww * Et)])
    return e0


# ***funcao 13***
def calcular_ilum_faces(m, D0, i1, alpha, beta):
    ca = math.cos(alpha[m])
    sa = math.sin(alpha[m])
    cb = math.cos(beta[m])
    sb = math.sin(beta[m])
    T1 = []
    T1 = [[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]]
    # print(m, T1)
    T2 = []
    T2 = [[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]]
    Dzero = np.array(D0[i1])
    D1 = T1 * Dzero.transpose()
    D2 = T2 * D1


# funcao main
def main(name):
    start = timer()

    # print("PyPOFacets - v0.1")
    # print("=================")
    # freq = float(input("Entre com a frequencia do radar em Hz:"))
    freq = 15000000
    # print(freq)
    # print("Frequencia radar em Hz:", freq, "Hz")

    # ***funcao 1***
    waveL = calcular_comprimento_de_onda(freq)
    # print("Comprimento de onda em metros:", waveL, "m")
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
    ipol = 0

    # ***funcao 2***
    Et, Ep = calcular_polarizacao_onda_incidente(ipol)
    # Co = 1

    # ***funcao 3***
    xpts, ypts, zpts = ler_coordenadas_modelo(name)
    # print("Coordenadas x dos pontos do modelo: ", xpts)
    # print("Coordenadas y dos pontos do modelo: ", ypts)
    # print("Coordenadas z dos pontos do modelo: ", zpts)
    nverts = len(xpts)
    # print("Quantidade de vertices do modelo: ", nverts)

    # ***funcao 4***
    facets = ler_facets_modelo(name)
    # print("Informacoes das faces do modelo: ", facets)

    # ***funcao 5***
    nfcv, node1, node2, node3, ilum, Rs = gerar_matriz_transposta(facets)
    # print("Numeracao das faces do modelo no arquivo: ", nfcv)
    # print("Primeiro componente de cada face: ", node1)
    # print("Segundo componente de cada face: ", node2)
    # print("Terceiro componente de cada face: ", node3)
    ntria = len(node3)
    # print("Quantidade de faces do modelo: ", ntria)

    # ***funcao 6***
    r = gerar_coordenadas_pontos(xpts, ypts, zpts)
    # print("Coordenadas de cada vertice do modelo: ", r)
    # inicio plot

    # ***funcao 7***
    # plotar_grafico_3d_modelo(node1, node2, node3, r)

    # Oct 138 - Pattern Loop
    # pstart = float(input("Entre com o valor de referencia do angulo phi inicial em graus:"))
    pstart = 0
    # pstop = float(input("Entre com o valor de referencia do angulo phi final em graus:"))
    pstop = 0
    # delp = float(input("Entre com o passo do angulo phi em graus:"))
    delp = 0
    # tstart = float(input("Entre com o valor de referencia do angulo theta inicial em graus:"))
    tstart = 0
    # tstop = float(input("Entre com o valor de referencia do angulo theta final em graus:"))
    tstop = 360
    # delt = float(input("Entre com o passo do angulo theta em graus:"))
    # delt = 2
    delt = 90

    # ***funcao 8***
    it, ip, delp, delt = calcular_refs_geometria_modelo(pstart, pstop, delp, tstart, tstop, delt, rad)
    # print("last step")
    # Get edge vectors and normal from edge cross products - OctT 168

    # ***funcao 9***
    areai, beta, alpha = calcular_arestas_e_normal_triangulos(node1, node2, node3, r)

    phi = []
    theta = []
    D0 = []
    R = []
    e0 = []

    for i1 in range(0, int(ip)):
        for i2 in range(0, int(it)):

            # ***funcao 10***
            u, v, w, uu, vv, ww, sp, cp, D0 = calcular_angulos_globais_e_direcoes(i1, i2, ip, it, pstart, delp, rad, tstart, delt, phi, theta, D0)

            # ***funcao 11***
            R = calcular_vetor_unitario_radial_sistema_coord_esfericas(u, v, w, R)

            # ***funcao 12***
            e0 = calcular_campo_eletrico_incidente_coord_cartesianas_globais(uu, vv, ww, Et, Ep, sp, cp, e0)

            # Begin loop over triangles
            sumt = 0
            sump = 0
            sumdt = 0
            sumdp = 0
            #for m in range(ntria):
                # OctT 236
                # Test to see if front face is illuminated: AFAZER
                # Local direction cosines

                # ***funcao 13***
                # calcular_ilum_faces(m, D0, i1, alpha, beta)

    # print(np.transpose(e0))

    end = timer()
    print(end - start, "seg")


if __name__ == '__main__':

    # name = input("Entre com o nome do diretorio do modelo:")
    # name = "/home/clayton/Documentos/POFACETS/pofacets2.2nogui/PLATE"
    # name = "/home/clayton/Documentos/POFACETS/pofacets2.2nogu/BOX"
    name = "./BOX"
    main(name)
