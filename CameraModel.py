import numpy as np
from functions import getRotationMatrix

class Camera:
    
    def __init__(self, handler, position, euler_angles, resolution:tuple ,distortion = None):
        self.handler = handler
        self.pos_x, self.pos_y, self.pos_z = position
        self.alpha, self.beta, self.gamma = euler_angles
        
        self.K =  np.eye(3) if not distortion else distortion

        self.R = getRotationMatrix(self.alpha,self.beta,self.gamma)
        self.resolution = resolution

    def getCorrectRawImage(self):
        img = self.handler.raw_image()
        img = np.array(img,dtype=np.uint8)
        img.resize([self.resolution[0],self.resolution[1],3])
        return img

    def getRawImage(self):
        return self.handler.raw_image()
    
    def getCameraInfo(self):
        return self.handler.read()