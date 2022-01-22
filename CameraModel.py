import numpy as np
from camera_tools import *



class CameraModel():
		
	def __init__(self, position:np.ndarray, orientation:np.ndarray, resolution:float ,afov:float) -> None:

		self.pos_x, self.pos_y, self.pos_z = position
		self.alpha, self.beta, self.gamma = np.deg2rad(orientation)
		self.resolution = int(resolution) #TODO: add xres, yres
		self.afov = afov
		self.focal_length, self.P = make_camera_matrix(position,self.alpha, self.beta, self.gamma, resolution, afov)

	def project_point(self, point: np.ndarray) -> np.ndarray:
		res = self.P @ point
		return res/res[2]

	def invert_projection (self, image_point:np.ndarray, height:float) -> np.ndarray:
		M = self.P[:3,:3]
		p4 = self.P[:,3:]
		M_inv = np.linalg.inv(M)
		mu = (height + (M_inv@p4)[2][0])/((M_inv @ image_point)[2][0])
		return M_inv @ (mu*image_point - p4)


	def __str__(self):
		return f"	{[self.pos_x,self.pos_y,self.pos_z,]=}\n\
					{[self.alpha, self.beta, self.gamma]=}\n\
					{self.resolution=}\n\
					{self.afov=}\n\
					{self.focal_length=}\n\
					{self.P=}"