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


def read_arguments(argv):
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    program_name = argv[0]
    input_data_file = argv[2]
    input_model = argv[1]
    if len(argv) < 4:
        output_dir = os.path.join("output", "m5_chatgpt", str(time))
    else:
        output_dir = argv[3]
    return time, program_name, input_data_file, input_model, output_dir


def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


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


def calculate_polarization(ipol):
    if ipol == 0:
        et = 1 + 0j
        ep = 0 + 0j
    elif ipol == 1:
        et = 0 + 0j
        ep = 1 + 0j
    return et, ep


def read_coordinates(input_model):
    fname = input_model + "/coordinates.m"
    coordinates = np.loadtxt(fname)
    xpts = coordinates[:, 0]
    ypts = coordinates[:, 1]
    zpts = coordinates[:, 2]
    nverts = len(xpts)
    return coordinates, xpts, ypts, zpts, nverts


def read_facets(input_model):
    fname2 = input_model + "/facets.m"
    facets = np.loadtxt(fname2)
    return facets


def extract_facet_nodes(facets):
    node1 = facets[:, 1]
    node2 = facets[:, 2]
    node3 = facets[:, 3]
    return node1, node2, node3


def assemble_vertices(xpts, ypts, zpts, nverts):
    x = xpts
    y = ypts
    z = zpts
    r = [[x[i], y[i], z[i]]
         for i in range(nverts)]
    return r


def assemble_triangles(node1, node2, node3):
    ntria = len(node3)
    vind = [[node1[i], node2[i], node3[i]]
            for i in range(ntria)]
    return ntria, vind


def render_model_plot(r, vind, ntria, input_model, output_dir):
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


def prepare_angular_steps(pstart, pstop, delp, tstart, tstop, delt):
    if delp == 0:
        delp = 1
    if pstart == pstop:
        phr0 = pstart*RAD

    if delt == 0:
        delt = 1
    if tstart == tstop:
        thr0 = tstart*RAD

    it = math.floor((tstop-tstart)/delt)+1
    ip = math.floor((pstop-pstart)/delp)+1
    return delp, delt, it, ip


def build_output_header(now, program_name, input_data_file, input_model,
                        freq, corr, delstd, ipol, pstart, pstop, delp,
                        tstart, tstop, delt):
    r_data = [
            now, program_name, input_data_file, input_model,
            freq, corr, delstd, ipol, pstart, pstop,
            delp, tstart, tstop, delt
        ]
    header = '\n'.join(map(str, r_data)) + '\n'
    return header


def open_output_files(output_dir):
    fileR = open(os.path.join(output_dir, FILENAME_R), 'w')
    fileE0 = open(os.path.join(output_dir, FILENAME_E0), 'w')
    return fileR, fileE0


def write_output_headers(fileR, fileE0, header):
    fileR.write(header)
    fileE0.write(header)


def calculate_angles(phi, theta, i1, i2, pstart, delp, tstart, delt):
    phi.append(pstart + i1 * delp)
    phr = phi[i2] * RAD
    theta.append(tstart + i2 * delt)
    thr = theta[i2] * RAD
    return phr, thr


def calculate_trigonometric_components(phr, thr):
    st = math.sin(thr)
    ct = math.cos(thr)
    cp = math.cos(phr)
    sp = math.sin(phr)
    return st, ct, cp, sp


def calculate_direction_vector(st, ct, cp, sp):
    D0 = [st*cp, st*sp, ct]
    return D0


def calculate_polarization_basis(st, ct, cp, sp):
    E = [ct*cp, ct*sp, -st, sp, cp]
    return E


def write_direction_record(fileR, i2, D0):
    u, v, w = D0
    fileR.write(str(i2))
    fileR.write(" ")
    fileR.write(str([u, v, w]))
    fileR.write("\n")


def calculate_field_vector(E, et, ep):
    uu, vv, ww, sp, cp = E
    return [(uu * et - sp * ep), (vv * et + cp * ep), (ww * et)]


def write_field_record(fileE0, i2, field_vector):
    fileE0.write(str(i2))
    fileE0.write(" ")
    fileE0.write(str(field_vector))
    fileE0.write("\n")


def generate_output_records(fileR, fileE0, ip, it, pstart, delp,
                            tstart, delt, et, ep):
    phi = []
    theta = []
    R = []
    e0 = []
    for i1 in range(0, int(ip)):
        for i2 in range(0, int(it)):
            phr, thr = calculate_angles(phi, theta, i1, i2, pstart, delp,
                                        tstart, delt)
            st, ct, cp, sp = calculate_trigonometric_components(phr, thr)
            D0 = calculate_direction_vector(st, ct, cp, sp)
            E = calculate_polarization_basis(st, ct, cp, sp)
            write_direction_record(fileR, i2, D0)
            field_vector = calculate_field_vector(E, et, ep)
            write_field_record(fileE0, i2, field_vector)


def close_output_files(fileR, fileE0):
    fileR.close()
    fileE0.close()


if __name__ == "__main__":
    time, program_name, input_data_file, input_model, output_dir = read_arguments(sys.argv)
    create_output_directory(output_dir)
    freq, corr, delstd, ipol, pstart, pstop, delp, tstart, tstop, delt = read_parameters(input_data_file)
    waveL = calculate_wavelength(freq)
    et, ep = calculate_polarization(ipol)
    coordinates, xpts, ypts, zpts, nverts = read_coordinates(input_model)
    facets = read_facets(input_model)
    node1, node2, node3 = extract_facet_nodes(facets)
    r = assemble_vertices(xpts, ypts, zpts, nverts)
    ntria, vind = assemble_triangles(node1, node2, node3)
    now = render_model_plot(r, vind, ntria, input_model, output_dir)
    delp, delt, it, ip = prepare_angular_steps(pstart, pstop, delp, tstart, tstop, delt)
    header = build_output_header(now, program_name, input_data_file, input_model,
                                 freq, corr, delstd, ipol, pstart, pstop, delp,
                                 tstart, tstop, delt)
    fileR, fileE0 = open_output_files(output_dir)
    write_output_headers(fileR, fileE0, header)
    generate_output_records(fileR, fileE0, ip, it, pstart, delp, tstart,
                            delt, et, ep)
    close_output_files(fileR, fileE0)
