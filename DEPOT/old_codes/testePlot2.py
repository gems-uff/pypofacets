from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
fig1 = plt.figure()
ax = Axes3D(fig1)
x = [0, 1, 0, 0]
print(x)
y = [0, 0, 1, 0]
z = [0, 1, 1, 0]
# verts = [zip(x, y, z)]
ax.plot3D(x, y, z)
# fig2 = plt.figure()
# ax = Axes3D(fig2)
x1 = [1, 0, 1, 1]
print(x1[0])
print(isinstance(x1[0], int))
print(type(x1[0]))
print(type(x1))
y1 = [0, 1, 1, 0]
z1 = [1, 1, 1, 1]
ax.plot3D(x1, y1, z1)
# ax.add_collection3d(Poly3DCollection(verts))
ax.autoscale()
ax.set_title("Teste")
plt.show()
