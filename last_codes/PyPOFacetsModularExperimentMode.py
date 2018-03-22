import sys
import math
import numpy as np

from datetime import datetime


def read_param_input(input_data_file):
    params = open(input_data_file, 'r')
    param_list = []
    for line in params:
        if not line.startswith("#"):
            param_list.append(int(line))
    params.close()
    return param_list


def calculate_wavelength(freq):
    c = 3e8
    waveL = c / freq
    return waveL


def calculate_incident_wave_polarization(ipol):
    if ipol == 0:
        Et = 1 + 0j
        Ep = 0 + 0j
    elif ipol == 1:
        Et = 0 + 0j
        Ep = 1 + 0j
    return (Et, Ep)


def read_model_coordinates(input_model):
    fname = input_model + "/coordinates.m"
    coordinates = np.around(np.loadtxt(fname)).astype(int)
    return np.transpose(coordinates)


def read_facets_model(input_model):
    fname2 = input_model + "/facets.m"
    facets = np.around(np.loadtxt(fname2)).astype(int)
    return facets


def generate_transpose_matrix(facets):
    nfcv, node1, node2, node3, ilum, Rs = np.transpose(facets)
    return nfcv, node1, node2, node3, ilum, Rs


def generate_coordinates_points(xpts, ypts, zpts):
    r = list(zip(xpts, ypts, zpts))
    return r


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
    ip = math.floor((pstop-pstart)/delp)+1
    return it, ip, delp, delt


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
        d = [np.linalg.norm(A), np.linalg.norm(B), np.linalg.norm(C)]
        ss = 0.5*sum(d)
        areai.append(math.sqrt(ss*(ss-np.linalg.norm(A))*(ss-np.linalg.norm(B))*(ss-np.linalg.norm(C))))
        Nn = np.linalg.norm(N)
        N = N/Nn
        beta.append(math.acos(N[2]))
        alpha.append(math.atan2(N[1],N[0]))
    return areai, beta, alpha


def calculate_global_angles_and_directions(i1, i2, ip, it, pstart, delp, rad, tstart, delt, phi, theta, D0):
    phi.append(pstart + i1 * delp)
    phr = phi[i2] * rad
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
    U = u
    V = v
    W = w
    uu = ct * cp
    vv = ct * sp
    ww = -st
    return u, v, w, uu, vv, ww, sp, cp, D0


def calculate_spherical_coordinate_system_radial_unit_vector(fileR, i2, u, v, w, R):
    fileR.write(str(i2))
    fileR.write(" ")
    fileR.write(str([u, v, w]))
    fileR.write("\n")
    R.append([u, v, w])
    return R


def calculate_incident_field_in_global_cartesian_coordinates(fileE0, i2, uu, vv, ww, Et, Ep, sp, cp, e0):
    fileE0.write(str(i2))
    fileE0.write(" ")
    fileE0.write(str([(uu * Et - sp * Ep), (vv * Et + cp * Ep), (ww * Et)]))
    fileE0.write("\n")
    e0.append([(uu * Et - sp * Ep), (vv * Et + cp * Ep), (ww * Et)])
    return e0


def assemble_generate_output_file(corr, delp, delstd, delt, freq, ipol, pstart, pstop, tstart, tstop):
    phi = []
    theta = []
    D0 = []
    R = []
    e0 = []
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    filename_R = "R_PyPOFacetsModularExperimentMode_" + sys.argv[1] + "_" + sys.argv[2] + "_" + now + ".dat"
    filename_E0 = "E0_PyPOFacetsModularExperimentMode_" + sys.argv[1] + "_" + sys.argv[2] + "_" + now + ".dat"
    fileR = open(filename_R, 'w')
    fileE0 = open(filename_E0, 'w')
    r_data = [
        now, sys.argv[0], sys.argv[1], sys.argv[2],
        freq, corr, delstd, ipol, pstart, pstop,
        delp, tstart, tstop, delt
    ]
    text = '\n'.join(map(str, r_data)) + '\n'
    fileR.write(text)
    fileE0.write(text)
    for i1 in range(0, int(ip)):
        for i2 in range(0, int(it)):
            u, v, w, uu, vv, ww, sp, cp, D0 = calculate_global_angles_and_directions(i1, i2, ip, it, pstart, delp, rad,
                                                                                     tstart, delt, phi, theta, D0)
            R = calculate_spherical_coordinate_system_radial_unit_vector(fileR, i2, u, v, w, R)
            e0 = calculate_incident_field_in_global_cartesian_coordinates(fileE0, i2, uu, vv, ww, Et, Ep, sp, cp, e0)
    fileR.close()
    fileE0.close()
    return r_data


input_model = sys.argv[1]
input_data_file = sys.argv[2]

freq, corr, delstd, ipol, pstart, pstop, delp, tstart, tstop, delt = read_param_input(input_data_file)

waveL = calculate_wavelength(freq)

corel = corr / waveL
delsq = delstd ** 2
bk = 2 * math.pi / waveL
cfact1 = math.exp(-4 * bk ** 2 * delsq)
cfact2 = 4 * math.pi * (bk * corel) ** delsq
rad = math.pi / 180
Lt = 0.05
Nt = 5

Et, Ep = calculate_incident_wave_polarization(ipol)
Co = 1

xpts, ypts, zpts = read_model_coordinates(input_model)
nverts = len(xpts)

facets = read_facets_model(input_model)

nfcv, node1, node2, node3, ilum, Rs = generate_transpose_matrix(facets)
ntria = len(node3)

r = generate_coordinates_points(xpts, ypts, zpts)

it, ip, delp, delt = calculate_refs_geometry_model(pstart, pstop, delp, tstart, tstop, delt, rad)

areai, beta, alpha = calculate_edges_and_normal_triangles(node1, node2, node3, r)

result = assemble_generate_output_file(corr, delp, delstd, delt, freq, ipol, pstart, pstop, tstart, tstop)
