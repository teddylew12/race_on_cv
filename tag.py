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
    def add_tag(self,id,x,y,z,theta_x,theta_y,theta_z):

        self.locations[id]=self.inchesToTranslationVector(x,y,z)
        self.orientations[id]=self.eulerAnglesToRotationMatrix(theta_x,theta_y,theta_z)
    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self, theta_x,theta_y,theta_z):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta_x), -np.sin(theta_x)],
                        [0, np.sin(theta_x), np.cos(theta_x)]
                        ])

        R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                        [0, 1, 0],
                        [-np.sin(theta_y), 0, np.cos(theta_y)]
                        ])

        R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                        [np.sin(theta_z), np.cos(theta_z), 0],
                        [0, 0, 1]
                        ])

        R = np.matmul(R_z, np.matmul(R_y, R_x))

        return R.T
    def inchesToTranslationVector(self,x,y,z):
        return np.array([[x],[y],[z]])*0.0254
    def estimate_pose(self, tag_id, R,t):
        local = self.tag_corr @ R.T @ t
        return self.orientations[tag_id] @ local + self.locations[tag_id]

