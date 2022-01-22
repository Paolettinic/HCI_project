import numpy as np

cos = np.cos
sin = np.sin
tan = np.tan
atan2 = np.arctan2


def make_rotation_x(alpha) -> np.ndarray:
	return np.array([
		[1,		0,		  	0			],
		[0,		cos(alpha),	-sin(alpha)	],
		[0,		sin(alpha),	cos(alpha)	]
	])

def make_rotation_y(beta) -> np.ndarray:
	return np.array([
		[cos(beta),		0,		sin(beta)],
		[0,				1,				0],
		[-sin(beta),	0,		cos(beta)]
	])
def make_rotation_z(gamma) -> np.ndarray:
	return np.array([
		[cos(gamma),	-sin(gamma),	0],
		[sin(gamma),	cos(gamma),		0],
		[0,				0,				1]
	])
	

def make_rotation_matrix(alpha, beta, gamma) -> np.ndarray:
	'''
		Coppelia follows Euler's angle convention:
		R_tot = Rx(alpha) x Ry(beta) x Rz(gamma)
		also, the coordinate frame is read bottom-up,r to l -> another rotation around z
	'''
	rot = make_rotation_x(alpha) @ make_rotation_y(beta) @ make_rotation_z(gamma + np.pi)
	return rot

def make_intinsics_matrix(resolution:int, angle:float) -> np.ndarray:
		f = resolution/(2*tan(np.deg2rad(angle)/2))
		return f, np.array([
			[f,	0,	resolution/2,	0],
			[0,	f,	resolution/2,	0],
			[0,	0,	1,				0]
		])



def make_extrinsics_matrix(position:np.ndarray, alpha:float, beta:float, gamma:float) -> np.ndarray:
	rotation = make_rotation_matrix(alpha,beta,gamma)
	rotation_T = rotation.transpose()
	pos = np.array([position]).T
	em = np.block([
		[rotation_T, 		-rotation_T @ pos	] ,
		[np.zeros((1,3)),	1					]
	])
	return em


def make_camera_matrix(
		position: np.ndarray, 
		alpha: float, 
		beta: float,
		gamma: float, 
		resolution: int,
		angle: float
	) -> np.ndarray:

	f, intrinsics = make_intinsics_matrix(resolution,angle) 
	return f, intrinsics @ make_extrinsics_matrix(position, alpha, beta, gamma)

