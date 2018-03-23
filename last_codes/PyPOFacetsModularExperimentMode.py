import sys
import math
import numpy as np


from datetime import datetime
from contextlib import contextmanager

RAD = math.pi / 180


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
    return node1, node2, node3, ilum, Rs


def generate_coordinates_points(xpts, ypts, zpts):
    r = list(zip(xpts, ypts, zpts))
    return r


def calculate_refs_geometry_model(pstart, pstop, delp, tstart, tstop, delt):
    if delp == 0:
        delp = 1
    if pstart == pstop:
        phr0 = pstart * RAD
    if delt == 0:
        delt = 1
    if tstart == tstop:
        thr0 = tstart * RAD
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

def prepare_output(input_model, input_data_file, corr, delp, delstd, delt, freq, ipol, pstart, pstop, tstart, tstop):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    filename_R = "R_PyPOFacetsModularExperimentMode_" + input_model + "_" + input_data_file + "_" + now + ".dat"
    filename_E0 = "E0_PyPOFacetsModularExperimentMode_" + input_model + "_" + input_data_file + "_" + now + ".dat"
    filename_plot = "plot_PyPOFacetsModularExperimentMode_" + input_model + "_" + input_data_file + "_" + now + ".png"
    r_data = [
        now, sys.argv[0], sys.argv[1], sys.argv[2],
        freq, corr, delstd, ipol, pstart, pstop,
        delp, tstart, tstop, delt
    ]
    text = '\n'.join(map(str, r_data)) + '\n'
    return filename_R, filename_E0, filename_plot, text


def calculate_global_angles_and_directions(ip, it, pstart, delp, tstart, delt):
    i2s = []
    D0 = []
    phi = []
    theta = []
    E = []
    for i1 in range(0, int(ip)):
        phi.append([])
        theta.append([])
        for i2 in range(0, int(it)):
            i2s.append((i1, i2))
            phi[i1].append(pstart + i1 * delp)
            phr = phi[i1][i2] * RAD
            theta[i1].append(tstart + i2 * delt)
            thr = theta[i1][i2] * RAD
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
            E.append([uu, vv, ww, sp, cp])
    return i2s, D0, E, phi, theta




def calculate_spherical_coordinate_system_radial_unit_vector(i2s, D0, filename_R, common_data):
    with open(filename_R, "w") as fileR:
        fileR.write(common_data)
        for (i1, i2), elements in zip(i2s, D0):
            fileR.write(str(i2))
            fileR.write(" ")
            fileR.write(str(elements))
            fileR.write("\n")


def calculate_incident_field_in_global_cartesian_coordinates(i2s, E, Et, Ep, filename_E0, common_data):
    e0 = []
    with open(filename_E0, "w") as fileE0:
        fileE0.write(common_data)
        for (i1, i2), elements in zip(i2s, E):
            uu, vv, ww, sp, cp = elements
            incident_field = [(uu * Et - sp * Ep), (vv * Et + cp * Ep), (ww * Et)]
            e0.append(incident_field)
            fileE0.write(str(i2))
            fileE0.write(" ")
            fileE0.write(str(incident_field))
            fileE0.write("\n")
    return e0


def illuminate_faces(i2s, R, E, E0, node3, xpts, ypts, zpts, ilum, Rs, areai, alpha, beta, corr, waveL):
    """Not implemented"""
    corel = corr / waveL
    bk = 2 * math.pi / waveL
    Lt = 0.05
    Nt = 5
    Co = 1
    delsq = delstd ** 2
    cfact1 = math.exp(-4 * bk ** 2 * delsq)
    cfact2 = 4 * math.pi * (bk * corel) ** delsq
    D0 = R

    sth, sph = [], []

    for (i1, i2), d0, e, e0 in zip(i2s, R, E, E0):
        uu, vv, ww, sp, cp = e
        u, v, w = d0
        if len(sth) - 1 < i1:
            sth.append([])
            sph.append([])
        #for m in range(len(node3)):  # for m in mslice[1:ntria]
        #sth[i1].append(10 * ...)
        #sph[i2].append(10 * ...)
    return sth, sph

def plot_range(sth, sph, phi, theta, filename_plot):
    """Not implemented"""
    with open(filename_plot, "w"):
        pass



input_model = sys.argv[1]
input_data_file = sys.argv[2]


freq, corr, delstd, ipol, pstart, pstop, delp, tstart, tstop, delt = read_param_input(input_data_file)

waveL = calculate_wavelength(freq)

Et, Ep = calculate_incident_wave_polarization(ipol)

xpts, ypts, zpts = read_model_coordinates(input_model)

facets = read_facets_model(input_model)

node1, node2, node3, ilum, Rs = generate_transpose_matrix(facets)

points = generate_coordinates_points(xpts, ypts, zpts)

it, ip, delp, delt = calculate_refs_geometry_model(pstart, pstop, delp, tstart, tstop, delt)

areai, beta, alpha = calculate_edges_and_normal_triangles(node1, node2, node3, points)

filename_R, filename_E0, filename_plot, common_data = prepare_output(input_model, input_data_file, corr, delp, delstd, delt, freq, ipol, pstart, pstop, tstart, tstop)

i2s, D0, E, phi, theta = calculate_global_angles_and_directions(ip, it, pstart, delp, tstart, delt)
calculate_spherical_coordinate_system_radial_unit_vector(i2s, D0, filename_R, common_data)
E0 = calculate_incident_field_in_global_cartesian_coordinates(i2s, E, Et, Ep, filename_E0, common_data)
sth, sph = illuminate_faces(i2s, D0, E, E0, node3, xpts, ypts, zpts, ilum, Rs, areai, alpha, beta, corr, waveL)
plot_range(sth, sph, phi, theta, filename_plot)
