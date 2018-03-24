import sys
import math
import numpy as np

from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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


def calculate_incident_wave_polarization(ipol, waveL):
    if ipol == 0:
        et = 1 + 0j
        ep = 0 + 0j
    elif ipol == 1:
        et = 0 + 0j
        ep = 1 + 0j
    return (et, ep)


def read_model_coordinates(input_model):
    fname = input_model + "/coordinates.m"
    coordinates = np.around(np.loadtxt(fname)).astype(int)
    xpts = coordinates[:, 0]
    ypts = coordinates[:, 1]
    zpts = coordinates[:, 2]
    nverts = len(xpts)
    return xpts, ypts, zpts, nverts



def read_facets_model(input_model):
    fname2 = input_model + "/facets.m"
    facets = np.around(np.loadtxt(fname2)).astype(int)
    return facets


def generate_transpose_matrix(facets):
    node1 = facets[:, 1]
    node2 = facets[:, 2]
    node3 = facets[:, 3]
    return node1, node2, node3


def generate_coordinates_points(xpts, ypts, zpts, nverts):
    # r = list(zip(xpts, ypts, zpts))
    x = xpts
    y = ypts
    z = zpts
    r = [[x[i], y[i], z[i]]
         for i in range(nverts)]
    return r


def plot_model(node1, node2, node3, r):
    ntria = len(node3)
    vind = [[node1[i], node2[i], node3[i]]
            for i in range(ntria)]
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    for i in range(ntria):
        Xa = [int(r[int(vind[i][0]) - 1][0]), int(r[int(vind[i][1]) - 1][0]), int(r[int(vind[i][2]) - 1][0]),
              int(r[int(vind[i][0]) - 1][0])]
        Ya = [int(r[int(vind[i][0]) - 1][1]), int(r[int(vind[i][1]) - 1][1]), int(r[int(vind[i][2]) - 1][1]),
              int(r[int(vind[i][0]) - 1][1])]
        Za = [int(r[int(vind[i][0]) - 1][2]), int(r[int(vind[i][1]) - 1][2]), int(r[int(vind[i][2]) - 1][2]),
              int(r[int(vind[i][0]) - 1][2])]
        ax.plot3D(Xa, Ya, Za)
        ax.set_xlabel("X Axis")
    ax.set_title("3D Model: " + input_model)
    plt.savefig("plot_modular_pypofacets_" + now + ".png")
    plt.close()


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


def prepare_output(corr, delp, delstd, delt, freq, ipol, pstart, pstop, tstart, tstop):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    filename_R = "R_modular_pypofacets_" + now + ".dat"
    filename_E0 = "E0_modular_pypofacets_" + now + ".dat"
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
    return fileR, fileE0


def calculate_global_angles_and_directions(ip, it, pstart, delp, tstart, delt, phi, theta):
    i2s = []
    D0 = []
    # phi = []
    # theta = []
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
            uu = ct * cp
            vv = ct * sp
            ww = -st
            E.append([uu, vv, ww, sp, cp])
    return i2s, D0, E, phi, theta


def calculate_spherical_coordinate_system_radial_unit_vector(i2s, D0):
    for (i1, i2), elements in zip(i2s, D0):
        fileR.write(str(i2))
        fileR.write(" ")
        fileR.write(str(elements))
        fileR.write("\n")
    fileR.close()


def calculate_incident_field_in_global_cartesian_coordinates(i2s, E, Et, Ep):
    for (i1, i2), elements in zip(i2s, E):
        uu, vv, ww, sp, cp = elements
        incident_field = [(uu * Et - sp * Ep), (vv * Et + cp * Ep), (ww * Et)]
        fileE0.write(str(i2))
        fileE0.write(" ")
        fileE0.write(str(incident_field))
        fileE0.write("\n")
    fileE0.close()


input_data_file = sys.argv[2]

freq, corr, delstd, ipol, pstart, pstop, delp, tstart, tstop, delt = read_param_input(input_data_file)

waveL = calculate_wavelength(freq)

et, ep = calculate_incident_wave_polarization(ipol, waveL)

input_model = sys.argv[1]

xpts, ypts, zpts, nverts = read_model_coordinates(input_model)

facets = read_facets_model(input_model)

node1, node2, node3 = generate_transpose_matrix(facets)

points = generate_coordinates_points(xpts, ypts, zpts, nverts)

plot_model(node1, node2, node3, points)

it, ip, delp, delt = calculate_refs_geometry_model(pstart, pstop, delp, tstart, tstop, delt)

fileR, fileE0 = prepare_output(corr, delp, delstd, delt, freq, ipol, pstart, pstop, tstart, tstop)

phi = []

theta = []

i2s, D0, E, phi, theta = calculate_global_angles_and_directions(ip, it, pstart, delp, tstart, delt, phi, theta)

calculate_spherical_coordinate_system_radial_unit_vector(i2s, D0)

calculate_incident_field_in_global_cartesian_coordinates(i2s, E, et, ep)
