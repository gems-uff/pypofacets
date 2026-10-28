import sys
import math
import numpy as np
import os

from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

RAD = math.pi / 180
FILENAME_R = "R.dat"
FILENAME_E0 = "E0.dat"
FILENAME_PLOT = "plot.png"


def read_command_line(argv):
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    program_name = argv[0]
    input_data_file = argv[2]
    input_model = argv[1]
    if len(argv) < 4:
        output_dir = os.path.join("output", "monolithic", str(time))
    else:
        output_dir = argv[3]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return time, program_name, input_data_file, input_model, output_dir


def read_parameters(input_data_file):
    params = open(input_data_file, 'r')
    param_list = []
    for line in params:
        if not line.startswith("#"):
            param_list.append(int(line))
    freq, corr, delstd, ipol, pstart, pstop, delp, tstart, tstop, delt = param_list
    params.close()
    return freq, corr, delstd, ipol, pstart, pstop, delp, tstart, tstop, delt


def calculate_wavelength(freq):
    c = 3e8
    waveL = c / freq
    return waveL


def select_polarization(ipol):
    if ipol == 0:
        et = 1 + 0j
        ep = 0 + 0j
    elif ipol == 1:
        et = 0 + 0j
        ep = 1 + 0j
    return et, ep


def load_coordinates(input_model):
    fname = input_model + "/coordinates.m"
    coordinates = np.loadtxt(fname)
    return coordinates


def extract_coordinate_columns(coordinates):
    xpts = coordinates[:, 0]
    ypts = coordinates[:, 1]
    zpts = coordinates[:, 2]
    return xpts, ypts, zpts


def count_vertices(xpts):
    nverts = len(xpts)
    return nverts


def load_facets(input_model):
    fname2 = input_model + "/facets.m"
    facets = np.loadtxt(fname2)
    return facets


def extract_facet_columns(facets):
    node1 = facets[:, 1]
    node2 = facets[:, 2]
    node3 = facets[:, 3]
    return node1, node2, node3


def build_vertex_list(x, y, z, nverts):
    r = [[x[i], y[i], z[i]]
         for i in range(nverts)]
    return r


def build_facet_list(node1, node2, node3):
    ntria = len(node3)
    vind = [[node1[i], node2[i], node3[i]]
            for i in range(ntria)]
    return ntria, vind


def create_model_plot(r, vind, ntria, input_model, output_dir):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    for i in range(ntria):
        Xa = [int(r[int(vind[i][0])-1][0]), int(r[int(vind[i][1])-1][0]), int(r[int(vind[i][2])-1][0]), int(r[int(vind[i][0])-1][0])]
        Ya = [int(r[int(vind[i][0])-1][1]), int(r[int(vind[i][1])-1][1]), int(r[int(vind[i][2])-1][1]), int(r[int(vind[i][0])-1][1])]
        Za = [int(r[int(vind[i][0])-1][2]), int(r[int(vind[i][1])-1][2]), int(r[int(vind[i][2])-1][2]), int(r[int(vind[i][0])-1][2])]
        ax.plot3D(Xa, Ya, Za)
        ax.set_xlabel("X Axis")
    ax.set_title("3D Model: " + input_model)
    plt.savefig(os.path.join(output_dir, FILENAME_PLOT))
    plt.close()
    return now


def normalize_angle_steps(delp, pstart, pstop, delt, tstart, tstop):
    if delp == 0:
        delp = 1
    if pstart == pstop:
        phr0 = pstart*RAD

    if delt == 0:
        delt = 1
    if tstart == tstop:
        thr0 = tstart*RAD
    return delp, delt


def calculate_iteration_counts(tstart, tstop, delt, pstart, pstop, delp):
    it = math.floor((tstop-tstart)/delt)+1
    ip = math.floor((pstop-pstart)/delp)+1
    return it, ip


def build_header(now, program_name, input_data_file, input_model, freq, corr,
                 delstd, ipol, pstart, pstop, delp, tstart, tstop, delt):
    r_data = [
        now, program_name, input_data_file, input_model,
        freq, corr, delstd, ipol, pstart, pstop,
        delp, tstart, tstop, delt
    ]
    header = '\n'.join(map(str, r_data)) + '\n'
    return header


def open_output_files(output_dir, header):
    fileR = open(os.path.join(output_dir, FILENAME_R), 'w')
    fileE0 = open(os.path.join(output_dir, FILENAME_E0), 'w')
    fileR.write(header)
    fileE0.write(header)
    return fileR, fileE0


def append_scan_angles(phi, theta, pstart, delp, tstart, delt, i1, i2):
    phi.append(pstart + i1 * delp)
    phr = phi[i2] * RAD
    theta.append(tstart + i2 * delt)
    thr = theta[i2] * RAD
    return phi, theta, phr, thr


def calculate_direction_components(thr, phr):
    st = math.sin(thr)
    ct = math.cos(thr)
    cp = math.cos(phr)
    sp = math.sin(phr)
    D0 = [st*cp, st*sp, ct]
    E = [ct*cp, ct*sp, -st, sp, cp]
    return D0, E


def write_direction_records(fileR, fileE0, i2, D0, E, et, ep):
    u, v, w = D0
    fileR.write(str(i2))
    fileR.write(" ")
    fileR.write(str([u, v, w]))
    fileR.write("\n")
    uu, vv, ww, sp, cp = E
    fileE0.write(str(i2))
    fileE0.write(" ")
    fileE0.write(str([(uu * et - sp * ep), (vv * et + cp * ep), (ww * et)]))
    fileE0.write("\n")


def calculate_and_write_scan(fileR, fileE0, pstart, delp, tstart, delt,
                             ip, it, et, ep):
    phi = []
    theta = []
    R = []
    e0 = []
    for i1 in range(0, int(ip)):
        for i2 in range(0, int(it)):
            phi, theta, phr, thr = append_scan_angles(
                phi, theta, pstart, delp, tstart, delt, i1, i2
            )
            D0, E = calculate_direction_components(thr, phr)
            write_direction_records(fileR, fileE0, i2, D0, E, et, ep)
    fileR.close()
    fileE0.close()


def main():
    argv = sys.argv
    time, program_name, input_data_file, input_model, output_dir = read_command_line(argv)
    freq, corr, delstd, ipol, pstart, pstop, delp, tstart, tstop, delt = read_parameters(input_data_file)
    waveL = calculate_wavelength(freq)
    et, ep = select_polarization(ipol)
    coordinates = load_coordinates(input_model)
    xpts, ypts, zpts = extract_coordinate_columns(coordinates)
    nverts = count_vertices(xpts)
    facets = load_facets(input_model)
    node1, node2, node3 = extract_facet_columns(facets)
    x = xpts
    y = ypts
    z = zpts
    r = build_vertex_list(x, y, z, nverts)
    ntria, vind = build_facet_list(node1, node2, node3)
    now = create_model_plot(r, vind, ntria, input_model, output_dir)
    delp, delt = normalize_angle_steps(delp, pstart, pstop, delt, tstart, tstop)
    it, ip = calculate_iteration_counts(tstart, tstop, delt, pstart, pstop, delp)
    header = build_header(
        now, program_name, input_data_file, input_model, freq, corr, delstd,
        ipol, pstart, pstop, delp, tstart, tstop, delt
    )
    fileR, fileE0 = open_output_files(output_dir, header)
    calculate_and_write_scan(
        fileR, fileE0, pstart, delp, tstart, delt, ip, it, et, ep
    )


main()
