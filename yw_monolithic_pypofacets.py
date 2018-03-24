
import sys
import math
import numpy as np

from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

RAD = math.pi / 180

# @begin yw_monolithic_pypofacets
# @in  input_model  @as InputModel
# @in  input_data_file  @as InputDataFileName
# @in  fname @as CoordinatesFile  @URI file:{InputModel}/coordinates.m
# @in  fname2 @as FacetsFile  @URI file:{InputModel}/facets.m
# @in  fname3 @as InputDataFile  @URI file:{InputDataFileName}
# @out r_file @as R_Output  @URI file:{R_yw_monolithic_pypofacets_{Timestamp}.dat}
# @out e0_file @as E0_Output  @URI file:{E0_yw_monolithic_pypofacets_{Timestamp}.dat}
# @out plot_file @as PlotOutput  @URI file:{plot_yw_monolithic_pypofacets_{Timestamp}.png}


# @begin ReadParamInput
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
# @end ReadParamInput


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
# @out node1 @as Node2
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
# @out plot_file @as PlotOutput  @URI file:{plot_yw_monolithic_pypofacets_{Timestamp}.png}
ntria = len(node3)
vind = [[node1[i], node2[i], node3[i]]
        for i in range(ntria)]
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
plt.savefig("plot_yw_monolithic_pypofacets_" + now + ".png")
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
# @in  freq @as Freq
# @in  corr @as Corr
# @in  delstd @as Delstd
# @in  ipol @as InputPolarization
# @in  pstart @as PStart
# @in  pstop @as PStop
# @in  delp @as InputDelP
# @in  tstart @as TStart
# @in  tstop @as TStop
# @in  delt @as InputDelT
# @out r_file @as R_Output  @URI file:{R_yw_monolithic_pypofacets_{Timestamp}.dat}
# @out e0_file @as E0_Output  @URI file:{E0_yw_monolithic_pypofacets_{Timestamp}.dat}
filename_R = "R_yw_monolithic_pypofacets_"+now+".dat"
filename_E0 = "E0_yw_monolithic_pypofacets_"+now+".dat"
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
        # @in  phi @as Phi
        # @in  theta @as Theta
        # @out u @as U
        # @out v @as V
        # @out w @as W
        # @out uu @as UU
        # @out vv @as VV
        # @out ww @as WW
        # @out sp @as SP
        # @out cp @as CP
        # @out phi @as Phi
        # @out theta @as Theta
        phi.append(pstart + i1 * delp)
        phr = phi[i2] * RAD
        theta.append(tstart + i2 * delt)
        thr = theta[i2] * RAD
        st = math.sin(thr)
        ct = math.cos(thr)
        cp = math.cos(phr)
        sp = math.sin(phr)
        u = st*cp
        v = st*sp
        w = ct
        uu = ct*cp
        vv = ct*sp
        ww = -st
        # @end CalculateGlobalAnglesAndDirections

        # @begin CalculateSphericalCoordinateSystemRadialUnitVector
        # @in  ip @as IP
        # @in  it @as IT
        # @in  u @as U
        # @in  v @as V
        # @in  w @as W
        # @in  r_file @as R_Output  @URI file:{R_yw_monolithic_pypofacets_{Timestamp}.dat}
        # @out r_file @as R_Output  @URI file:{R_yw_monolithic_pypofacets_{Timestamp}.dat}
        fileR.write(str(i2))
        fileR.write(" ")
        fileR.write(str([u, v, w]))
        fileR.write("\n")
        # @end CalculateSphericalCoordinateSystemRadialUnitVector

        # @begin CalculateIncidentFieldInGlobalCartesianCoordinates
        # @in  ip @as IP
        # @in  it @as IT
        # @in  uu @as UU
        # @in  vv @as VV
        # @in  ww @as WW
        # @in  ep @as Et
        # @in  ep @as Ep
        # @in  sp @as SP
        # @in  cp @as CP
        # @in e0_file @as E0_Output  @URI file:{E0_yw_monolithic_pypofacets_{Timestamp}.dat}
        # @out e0_file @as E0_Output  @URI file:{E0_yw_monolithic_pypofacets_{Timestamp}.dat}
        fileE0.write(str(i2))
        fileE0.write(" ")
        fileE0.write(str([(uu * et - sp * ep), (vv * et + cp * ep), (ww * et)]))
        fileE0.write("\n")
        # @end CalculateIncidentFieldInGlobalCartesianCoordinates

fileR.close()
fileE0.close()

# @end yw_monolithic_pypofacets
