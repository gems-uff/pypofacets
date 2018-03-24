
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


# @begin yw_monolithic_pypofacets
# @in  argv  @as Arguments
# @in  fname @as CoordinatesFile  @URI file:{InputModel}/coordinates.m
# @in  fname2 @as FacetsFile  @URI file:{InputModel}/facets.m
# @in  fname3 @as InputDataFile  @URI file:{InputDataFileName}
# @out r_file @as R_Output  @URI file:{{OutputDir}/R.dat}
# @out e0_file @as E0_Output  @URI file:{{OutputDir}/E0.dat}
# @out plot_file @as PlotOutput  @URI file:{{OutputDir}/plot.png}


# @begin ReadArgs
# @in  argv  @as Arguments
# @out time  @as Time
# @out program_name  @as ProgramName
# @out input_model  @as InputModel
# @out input_data_file  @as InputDataFileName
# @out output_dir  @as OutputDir
argv = sys.argv
time = datetime.now().strftime("%Y%m%d%H%M%S")
program_name = argv[0]
input_data_file = argv[2]
input_model = argv[1]
if len(argv) < 4:
    output_dir = os.path.join("output", "yw_monolithic", str(time))
else:
    output_dir = argv[3]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# @end ReadArgs


# @begin ReadDataFileInput
# @in  input_data_file  @as InputDataFileName
# @in  fname3 @as InputDataFile  @URI file:{input_data_file}
# @out  freq @as Freq
# @out  corr @as Corr
# @out  delstd @as Delstd
# @out  ipol @as InputPolarization
# @out  pstart @as PStart
# @out  pstop @as PStop
# @out  delp @as InputDelP
# @out  tstart @as TStart
# @out  tstop @as TStop
# @out  delt @as InputDelT
input_data_file = sys.argv[2]
params = open(input_data_file, 'r')
param_list = []
for line in params:
    if not line.startswith("#"):
        param_list.append(int(line))
freq, corr, delstd, ipol, pstart, pstop, delp, tstart, tstop, delt = param_list
params.close()
# @end ReadDataFileInput


# @begin CalculateWaveLength
# @in  freq @as Freq
# @out waveL @as WaveLength
c = 3e8
waveL = c / freq
# @end CalculateWaveLength


# @begin CalculateIncidentWavePolarization
# @in ipol @as InputPolarization
# @in waveL @as WaveLength
# @out et @as Et
# @out ep @as Ep
if ipol == 0:
    et = 1 + 0j
    ep = 0 + 0j
elif ipol == 1:
    et = 0 + 0j
    ep = 1 + 0j
# @end CalculateIncidentWavePolarization


# @begin ReadModelCoordinates
# @in  input_model @as InputModel
# @in  fname @as CoordinatesFile  @URI file:{input_model}/coordinates.m
# @out xpts @as XPoints
# @out ypts @as YPoints
# @out zpts @as ZPoints
# @out nverts @as Nverts
input_model = sys.argv[1]
fname = input_model + "/coordinates.m"
coordinates = np.loadtxt(fname)
xpts = coordinates[:, 0]
ypts = coordinates[:, 1]
zpts = coordinates[:, 2]
nverts = len(xpts)
# @end ReadModelCoordinates


# @begin ReadFacetsModel
# @in  input_model @as InputModel
# @in  fname2 @as FacetsFile  @URI file:{input_model}/facets.m
# @out facets @as Facets
fname2 = input_model + "/facets.m"
facets = np.loadtxt(fname2)
# @end ReadFacetsModel


# @begin GenerateTransposeMatrix
# @in  facets @as Facets
# @out node1 @as Node1
# @out node2 @as Node2
# @out node3 @as Node3
node1 = facets[:, 1]
node2 = facets[:, 2]
node3 = facets[:, 3]
# @end GenerateTransposeMatrix


# @begin GenerateCoordinatesPoints
# @in  xpts @as XPoints
# @in  ypts @as YPoints
# @in  zpts @as ZPoints
# @in nverts @as Nverts
# @out r @as Points
x = xpts
y = ypts
z = zpts
r = [[x[i], y[i], z[i]]
     for i in range(nverts)]
# @end GenerateCoordinatesPoints


# @begin PlotModel
# @in  node1 @as Node1
# @in  node1 @as Node2
# @in  node3 @as Node3
# @in  r @as Points
# @in  output_dir @as OutputDir
# @out plot_file @as PlotOutput  @URI file:{{OutputDir}/plot.png}
ntria = len(node3)
vind = [[node1[i], node2[i], node3[i]]
        for i in range(ntria)]
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
# @end PlotModel


# @begin CalculateRefsGeometryModel
# @in  pstart @as PStart
# @in  pstop @as PStop
# @in  delp @as InputDelP
# @in  tstart @as TStart
# @in  tstop @as TStop
# @in  delt @as InputDelT
# @out it @as IT
# @out ip @as IP
# @out delp @as DelP
# @out delt @as DelT
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
# @end CalculateRefsGeometryModel


# @begin PrepareOutput
# @in  time @as Time
# @in  program_name @as ProgramName
# @in  input_data_file @as InputDataFileName
# @in  input_model @as InputModel
# @in  output_dir @as OutputDir
# @in  corr @as Corr
# @in  delp @as InputDelP
# @in  delstd @as Delstd
# @in  delt @as InputDelT
# @in  freq @as Freq
# @in  ipol @as InputPolarization
# @in  pstart @as PStart
# @in  pstop @as PStop
# @in  tstart @as TStart
# @in  tstop @as TStop
# @out e0_file @as E0_Output  @URI file:{{OutputDir}/E0.dat}
# @out r_file @as R_Output  @URI file:{{OutputDir}/R.dat}
r_data = [
        time, sys.argv[0], sys.argv[1], sys.argv[2],
        freq, corr, delstd, ipol, pstart, pstop,
        delp, tstart, tstop, delt
    ]
header = '\n'.join(map(str, r_data)) + '\n'
fileR = open(os.path.join(output_dir, FILENAME_R), 'w')
fileE0 = open(os.path.join(output_dir, FILENAME_E0), 'w')

fileE0.write(header)
fileR.write(header)
# @end PrepareOutput

phi = []
theta = []
R = []
e0 = []
for i1 in range(0, int(ip)):
    for i2 in range(0, int(it)):
        # @begin CalculateGlobalAnglesAndDirections
        # @in  ip @as IP
        # @in  it @as IT
        # @in  pstart @as PStart
        # @in  delp @as DelP
        # @in  tstart @as TStart
        # @in  delt @as DelT
        # @out D0 @as D0
        # @out E @as E
        phi.append(pstart + i1 * delp)
        phr = phi[i2] * RAD
        theta.append(tstart + i2 * delt)
        thr = theta[i2] * RAD
        st = math.sin(thr)
        ct = math.cos(thr)
        cp = math.cos(phr)
        sp = math.sin(phr)
        D0 = [st*cp, st*sp, ct]
        E = [ct*cp, ct*sp, -st, sp, cp]
        # @end CalculateGlobalAnglesAndDirections

        # @begin CalculateSphericalCoordinateSystemRadialUnitVector
        # @in  ip @as IP
        # @in  it @as IT
        # @in  D0 @as D0
        # @in  output_dir @as OutputDir
        # @out r_file @as R_Output  @URI file:{{OutputDir}/R.dat}
        fileR.write(str(i2))
        fileR.write(" ")
        fileR.write(str(D0))
        fileR.write("\n")
        # @end CalculateSphericalCoordinateSystemRadialUnitVector

        # @begin CalculateIncidentFieldInGlobalCartesianCoordinates
        # @in  ip @as IP
        # @in  it @as IT
        # @in  E @as E
        # @in  et @as Et
        # @in  ep @as Ep
        # @in  output_dir @as OutputDir
        # @out e0_file @as E0_Output  @URI file:{{OutputDir}/E0.dat}
        uu, vv, ww, sp, cp = E
        fileE0.write(str(i2))
        fileE0.write(" ")
        fileE0.write(str([(uu * et - sp * ep), (vv * et + cp * ep), (ww * et)]))
        fileE0.write("\n")
        # @end CalculateIncidentFieldInGlobalCartesianCoordinates

fileR.close()
fileE0.close()

# @end yw_monolithic_pypofacets
