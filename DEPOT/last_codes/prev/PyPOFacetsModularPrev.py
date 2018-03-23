# PyPOFacetsModular, Monostatic version
# Inspired by the software POFacets noGUI v2.2 in MATLAB

import math
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from timeit import default_timer as timer


# ***function 1***
def calculate_wavelength(freq):
    c = 3e8
    # print("speed of light: ", c)
    waveL = c / freq
    return waveL


# ***function 2***
def calculate_incident_wave_polarization(ipol):
    if ipol == 0:
        Et = 1 + 0j
        Ep = 0 + 0j
    elif ipol == 1:
        Et = 0 + 0j
        Ep = 1 + 0j
    else:
        print("error")
    return (Et, Ep)


# ***function 3***
def read_model_coordinates(name):
    fname = name + "/coordinates.m"
    # print("Path and file model coordinates: ", fname)
    coordinates = np.around(np.loadtxt(fname)).astype(int)
    # print("Model points coordinates: ", coordinates)
    return np.transpose(coordinates)


# ***function 4***
def read_facets_model(name):
    fname2 = name + "/facets.m"
    # print("Path and file facets model: ", fname2)
    facets = np.around(np.loadtxt(fname2)).astype(int)  #astype ver np.around(np.loadtxt(...))
    return facets


# ***function 5***
def generate_transpose_matrix(facets):
    nfcv, node1, node2, node3, ilum, Rs = np.transpose(facets)
    return nfcv, node1, node2, node3, ilum, Rs


# ***function 6***
def generate_coordinates_points(xpts, ypts, zpts):
    r = list(zip(xpts, ypts, zpts))
    return r


# ***function 7***
def plot_3d_graph_model(node1, node2, node3, r):
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    for point in zip(node1, node2, node3):
        Xa = [r[(point[0]) - 1][0], r[(point[1]) - 1][0], r[(point[2]) - 1][0], r[(point[0]) - 1][0]]
        # print(Xa)
        Ya = [r[point[0] - 1][1], r[point[1] - 1][1], r[point[2] - 1][1], r[point[0] - 1][1]]
        Za = [r[point[0] - 1][2], r[point[1] - 1][2], r[point[2] - 1][2], r[point[0] - 1][2]]
        # ax.plot_wireframe(Xa, Ya, Za)
        ax.plot(Xa, Ya, Za)
        ax.set_xlabel("testx")
    ax.set_title("test")
    plt.show()
    plt.close()


# ***function 8***
def calculate_refs_geometry_model(pstart, pstop, delp, tstart, tstop, delt, rad):
    if delp == 0:
        delp = 1
    if pstart == pstop:
        phr0 = pstart*rad

    if delt == 0:
        delt = 1
    if tstart == tstop:
        thr0 = tstart*rad

    it = math.floor((tstop-tstart)/delt)+1
    print("Number of horizontal rotations in the simulation: ", it)
    ip = math.floor((pstop-pstart)/delp)+1
    print("Number of vertical rotations in the simulation: ", ip)

    return it, ip, delp, delt


# ***function 9***
def calculate_edges_and_normal_triangles(node1, node2, node3, r):
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


# ***function 10***
def calculate_global_angles_and_directions(i1, i2, ip, it, pstart, delp, rad, tstart, delt, phi, theta, D0):
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


# ***function 11***
def calculate_spherical_coordinate_system_radial_unit_vector(u, v, w, R):
    R.append([u, v, w])
    return R


# ***function 12***
def calculate_incident_field_in_global_cartesian_coordinates(uu, vv, ww, Et, Ep, sp, cp, e0):
    e0.append([(uu * Et - sp * Ep), (vv * Et + cp * Ep), (ww * Et)])
    return e0


# ***function 13***
def calculate_ilum_faces(m, D0, i1, alpha, beta):
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


# function main
def main(name):
    start = timer()

    # print("PyPOFacets - v0.1")
    # print("=================")
    # freq = float(input("Enter the radar frequency in Hz:"))
    freq = 15000000
    # print(freq)
    # print("The radar frequency in Hz:", freq, "Hz")

    # ***function 1***
    waveL = calculate_wavelength(freq)
    # print("Wavelength in meters:", waveL, "m")
    # corr = float(input("Enter the correlation distance in meters:"))
    # corr = 0
    # corel = corr / wave
    # delstd = float(input("Enter the standard deviation in meters:"))
    # delstd = 0
    # delsq = delstd ** 2
    # bk = 2 * math.pi / wave
    # cfact1 = math.exp(-4 * bk ** 2 * delsq)
    # cfact2 = 4 * math.pi * (bk * corel) ** delsq
    rad = math.pi / 180
    # Lt = 0.05
    # Nt = 5
    # ipol = float(input("Enter the incident wave polarization:"))
    ipol = 0

    # ***function 2***
    Et, Ep = calculate_incident_wave_polarization(ipol)
    # Co = 1

    # ***function 3***
    xpts, ypts, zpts = read_model_coordinates(name)
    # print("Coordinates x (model points): ", xpts)
    # print("Coordinates y (model points): ", ypts)
    # print("Coordinates z (model points): ", zpts)
    nverts = len(xpts)
    # print("Number of model vertices: ", nverts)

    # ***function 4***
    facets = read_facets_model(name)
    # print("Model faces information: ", facets)

    # ***function 5***
    nfcv, node1, node2, node3, ilum, Rs = generate_transpose_matrix(facets)
    # print("Numbering of the model faces in the file: ", nfcv)
    # print("First component of each face: ", node1)
    # print("Second component of each face: ", node2)
    # print("Third component of each face: ", node3)
    ntria = len(node3)
    # print("Number of model faces: ", ntria)

    # ***function 6***
    r = generate_coordinates_points(xpts, ypts, zpts)
    # print("Coordinates of each vertex of the model: ", r)
    # start plot

    # ***function 7***
    plot_3d_graph_model(node1, node2, node3, r)

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
    delt = 360

    # ***function 8***
    it, ip, delp, delt = calculate_refs_geometry_model(pstart, pstop, delp, tstart, tstop, delt, rad)
    # print("last step")
    # Get edge vectors and normal from edge cross products - OctT 168

    # ***function 9***
    areai, beta, alpha = calculate_edges_and_normal_triangles(node1, node2, node3, r)

    phi = []
    theta = []
    D0 = []
    R = []
    e0 = []

    for i1 in range(0, int(ip)):
        for i2 in range(0, int(it)):

            # ***function 10***
            u, v, w, uu, vv, ww, sp, cp, D0 = calculate_global_angles_and_directions(i1, i2, ip, it, pstart, delp, rad, tstart, delt, phi, theta, D0)

            # ***function 11***
            R = calculate_spherical_coordinate_system_radial_unit_vector(u, v, w, R)

            # ***function 12***
            e0 = calculate_incident_field_in_global_cartesian_coordinates(uu, vv, ww, Et, Ep, sp, cp, e0)

            # Begin loop over triangles
            sumt = 0
            sump = 0
            sumdt = 0
            sumdp = 0
            for m in range(ntria):
                # OctT 236
                # Test to see if front face is illuminated: FUT
                # Local direction cosines
                print()
                # ***function 13***
                ###calculate_ilum_faces(m, D0, i1, alpha, beta)

    # print(np.transpose(e0))

    end = timer()
    print(end - start, "seg")


if __name__ == '__main__':

    # name = input("Entre com o nome do diretorio do modelo:")
    # name = "/home/clayton/Documentos/POFACETS/pofacets2.2nogui/PLATE"
    # name = "/home/clayton/Documentos/POFACETS/pofacets2.2nogu/BOX"
    name = "./BOX"
    # name = "/home/clayton/PycharmProjects/PyPOFacets/BOX"
    main(name)
