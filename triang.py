import matplotlib.tri as tri
import matplotlib.pyplot as plt
import numpy as np

# First create the x and y coordinates of the points.
plt.plot((0,2),(0,0), 'r')
plt.plot((0,2),(1,1), 'r')
ang = np.linspace(-np.pi/2, np.pi/2, 100)
xright = 2 + 0.5*np.cos(ang)
yright = 0.5 + 0.5*np.sin(ang)
xleft = -0.5*np.cos(ang)
yleft = 0.5 + 0.5*np.sin(ang)
plt.plot(xright, yright, 'r')
plt.plot(xleft, yleft, 'r')

xgrid = [0,1,2]
ygrid = [0, 0.5, 1]
grid = np.meshgrid(xgrid, ygrid)
grid_x = np.append(grid[0], [0.5*np.cos(np.pi-np.pi/6), 0.5*np.cos(np.pi+np.pi/6), 2 + 0.5*np.cos(np.pi/6), 2 + 0.5*np.cos(np.pi/6) ])
grid_y = np.append(grid[1], [0.5 + 0.5*np.sin(np.pi-np.pi/6), 0.5 + 0.5*np.sin(np.pi+np.pi/6), 0.5 + 0.5*np.sin(np.pi/6), 0.5 - 0.5*np.sin(np.pi/6) ])
plt.scatter(grid_x, grid_y)

plt.show()


# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = tri.Triangulation(grid_x, grid_y)

# Mask off unwanted triangles.

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.triplot(triang, 'bo-', lw=1)
ax1.set_title('triplot of Delaunay triangulation')
plt.show()