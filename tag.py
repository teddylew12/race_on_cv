import numpy as np


class Tag():
    def __init__(self, tag_size, family):
        self.family = family
        self.size = tag_size
        self.locations = {}
        self.orientations = {}
        corr = np.eye(3)
        corr[0, 0] = -1
        self.tag_corr = corr

    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self, theta):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta[0]), -np.sin(theta[0])],
                        [0, np.sin(theta[0]), np.cos(theta[0])]
                        ])

        R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                        [0, 1, 0],
                        [-np.sin(theta[1]), 0, np.cos(theta[1])]
                        ])

        R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                        [np.sin(theta[2]), np.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.matmul(R_z, np.matmul(R_y, R_x))

        return R.T

    def transform_to_global_frame(self, tag_id, R,t):
        local = self.tag_corr @ R.T @ t
        return self.orientations[tag_id] @ local + self.locations[tag_id]
    def inches_to_cm(self,x,y,z):
        return np.array([[x],[y],[z]])*0.0254
