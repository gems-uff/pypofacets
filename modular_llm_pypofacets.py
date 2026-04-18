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

def initialize_environment(argv):
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    program_name = argv[0]
    input_model = argv[1]
    input_data_file = argv[2]
    
    if len(argv) < 4:
        output_dir = os.path.join("output", "modular", str(time))
    else:
        output_dir = argv[3]
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    return program_name, input_model, input_data_file, output_dir

def load_parameters(input_data_file):
    params = open(input_data_file, 'r')
    param_list = []
    for line in params:
        if not line.startswith("#"):
            param_list.append(int(line))
    params.close()
    return param_list

def calculate_physical_constants(freq, ipol):
    c = 3e8
    waveL = c / freq

    if ipol == 0:
        et = 1 + 0j
        ep = 0 + 0j
    elif ipol == 1:
        et = 0 + 0j
        ep = 1 + 0j
    return et, ep

def load_geometry_data(input_model):
    fname = input_model + "/coordinates.m"
    coordinates = np.loadtxt(fname)
    xpts = coordinates[:, 0]
    ypts = coordinates[:, 1]
    zpts = coordinates[:, 2]
    nverts = len(xpts)

    fname2 = input_model + "/facets.m"
    facets = np.loadtxt(fname2)
    
    node1 = facets[:, 1]
    node2 = facets[:, 2]
    node3 = facets[:, 3]
    
    return xpts, ypts, zpts, nverts, node1, node2, node3

def generate_model_plot(input_model, output_dir, xpts, ypts, zpts, nverts, node1, node2, node3):
    r = [[xpts[i], ypts[i], zpts[i]] for i in range(nverts)]
    ntria = len(node3)
    vind = [[node1[i], node2[i], node3[i]] for i in range(ntria)]
    
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

def setup_iteration_bounds(pstart, pstop, delp, tstart, tstop, delt):
    if delp == 0:
        delp = 1
    if pstart == pstop:
        phr0 = pstart * RAD

    if delt == 0:
        delt = 1
    if tstart == tstop:
        thr0 = tstart * RAD

    it = math.floor((tstop - tstart) / delt) + 1
    ip = math.floor((pstop - pstart) / delp) + 1
    return ip, it, delp, delt

def initialize_output_files(output_dir, program_name, input_data_file, input_model, param_list):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    r_data = [now, program_name, input_data_file, input_model] + param_list
    header = '\n'.join(map(str, r_data)) + '\n'
    
    fileR = open(os.path.join(output_dir, FILENAME_R), 'w')
    fileE0 = open(os.path.join(output_dir, FILENAME_E0), 'w')
    
    fileR.write(header)
    fileE0.write(header)
    return fileR, fileE0

def compute_and_save_fields(fileR, fileE0, ip, it, pstart, delp, tstart, delt, et, ep):
    phi = []
    theta = []
    for i1 in range(0, int(ip)):
        for i2 in range(0, int(it)):
            current_phi = pstart + i1 * delp
            phi.append(current_phi)
            phr = current_phi * RAD
            
            current_theta = tstart + i2 * delt
            theta.append(current_theta)
            thr = current_theta * RAD
            
            st = math.sin(thr)
            ct = math.cos(thr)
            cp = math.cos(phr)
            sp = math.sin(phr)
            
            D0 = [st * cp, st * sp, ct]
            E = [ct * cp, ct * sp, -st, sp, cp]
            
            u, v, w = D0
            fileR.write(str(i2) + " " + str([u, v, w]) + "\n")
            
            uu, vv, ww, sp_val, cp_val = E
            fileE0.write(str(i2) + " " + str([(uu * et - sp_val * ep), (vv * et + cp_val * ep), (ww * et)]) + "\n")

if __name__ == "__main__":
    program_name, input_model, input_data_file, output_dir = initialize_environment(sys.argv)
    
    param_list = load_parameters(input_data_file)
    freq, corr, delstd, ipol, pstart, pstop, delp, tstart, tstop, delt = param_list
    
    et, ep = calculate_physical_constants(freq, ipol)
    
    xpts, ypts, zpts, nverts, node1, node2, node3 = load_geometry_data(input_model)
    
    generate_model_plot(input_model, output_dir, xpts, ypts, zpts, nverts, node1, node2, node3)
    
    ip, it, adj_delp, adj_delt = setup_iteration_bounds(pstart, pstop, delp, tstart, tstop, delt)
    
    fileR, fileE0 = initialize_output_files(output_dir, program_name, input_data_file, input_model, param_list)
    
    compute_and_save_fields(fileR, fileE0, ip, it, pstart, adj_delp, tstart, adj_delt, et, ep)
    
    fileR.close()
    fileE0.close()