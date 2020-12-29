import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
center=np.array([[.75],[1.5],[1.]])
total_time=10
w=2*np.pi/total_time
pos=np.empty(shape=(3,300),dtype=np.float)
for i,t in enumerate(np.linspace(0.,total_time,300)):
    pos[:,i:i+1]=center + np.array([[np.cos(t*w)],[np.sin(t*4*w)],[np.sin(t*w)]])

positions=pos
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(25, 10)
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
ax.set_title('Positions')
ax.scatter(positions[0,:], positions[2,:], positions[1,:], color="b")
plt.show()


