# import numpy as np
# import os
# from CameraModel import CameraModel

# np.set_printoptions(precision=3,suppress=True)
# if __name__ == "__main__":
# 	point = np.array([[253],[100],[1]])
# 	pos = np.array([[5],[5],[1.67],[1]])
# 	cm = CameraModel(
# 		"Vs1",
# 		np.array([9.35,9.35,2.2]),
# 		np.array([129.23, -37.76, 26.57]),
# 		512.0,
# 		75.0		
# 	)
# 	print(cm)

# 	print(cm.project_point(pos))
# 	print(cm.invert_projection(point,1.67))

import numpy as np

def merge_single_to_cluster(single,avg_cluster,maxdiff):
	tmp_single = single.copy()
	tmp_cluster = avg_cluster.copy()



def merge_cluster(avg_array,maxdiff):
	tmp = avg_array.copy()
	r = np.arange(len(avg_array))
	ret = []
	while len(tmp):
		seed = tmp[0]
		mask = np.linalg.norm(tmp - seed,axis = 1) <= maxdiff
		ret.append(r[mask])
		tmp = tmp[~mask]
		r = r[~mask]
	return ret

def points_cluster(array, maxdiff):
	if len(array) > 0:
		assert len(array[0]) >= 2
		tmp = array.copy()
		cluster , single = [], []
		while len(tmp):
			# select seed
			seed = tmp[0]
			mask = np.linalg.norm(tmp - seed,axis=1) <= maxdiff
			if np.count_nonzero(mask) > 1: 
				print("Distance Violated!")
				cluster.append(tmp[mask])
			else:
				single.append(tmp[mask][0])
			tmp = tmp[~mask]
		return cluster,single
	else:
		return [],[]

if __name__ == "__main__":
	d_c , d_s = [], []
	b = np.array([
		[2. , 1. ],
		# [2. , 1.1],
		# [5	,	0],
		# [1	, 	0],
		# [1.1,   0],
		[2, 3],
		[2.1,3]
	])
	f = np.array([
		[2.09 , 0.9 ],
		[2.01 , 1.1],
		# [6,	0],
		# [5.1,0],
		# [1	, 	0],
		# [1.1,   0],
	])

	c,s = points_cluster(b,0.3)
	d_c += c
	d_s += s
	c,s = points_cluster(f,0.3)
	d_c += c
	d_s += s
	d_s = np.array(d_s)
	d_c = np.array(d_c)
	print(d_s)
	print("_"*13)

	print(d_c)
	print("_"*13)
	points = []
	if len(d_s) > 0:
		indices = merge_cluster(d_s,0.5)
		for idx in indices:
			points.append(np.average(d_s[idx],axis = 0))
	if len(d_c) > 0:
		avg_c = np.array([np.average(a,axis = 0) for a in d_c])
		indices = merge_cluster(avg_c,0.5)
		print("-"*13)
		print(indices)
		for idx in indices:
			# points.append(np.array([ar for ar in max(d_c[idx],key=len)]))
			for ar in max(d_c[idx],key=len):
				points.append(np.array(ar))
		print("."*13)
	print(points)

	# b = np.array([
	# 	[2. , 1. ],
    # 	[2. , 1.1],
    # 	[2.1, 1.1],
    # 	[3. , 0. ],
	# 	[3. , 0.1],
	# 	[5	,	0]
	# ])
	# f = np.array([
	# 	[2.09 , 0.9 ],
    # 	[2.01 , 1.1],
    # 	[2.1, 1],
    # 	[3.01 , 0.01 ],
	# 	[3.01 , 0.11],
	# 	[3.01 , 0.1],
	# 	[5	,	0]
	# ])