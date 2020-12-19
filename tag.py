import numpy as np
class Tag():
    def __init__(self,tag_size,family):
        self.family=family
        self.size=tag_size
        self.locations={}
        self.orientations={}
    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self,theta) :
        R_x = np.array([[1,         0,                0                 ],
                        [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                        [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                        ])
     
        R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                        [0,                   1,      0                 ],
                        [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                        ])
                    
        R_z = np.array([[np.cos(theta[2]),   -np.sin(theta[2]),     0],
                        [np.sin(theta[2]),    np.cos(theta[2]),     0],
                        [0,                   0,                    1]
                        ])
              
        R = np.matmul(R_z, np.matmul( R_y, R_x ))

        return R