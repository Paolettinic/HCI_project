from CameraModel import CameraModel
import b0RemoteApi
import cv2
import numpy as np
import os
from detection_thread import ImageDetectionThread
from vision_sensor import VisionSensor
import time
from generator_model import Generator
from create_high_fps_video import create_high_fps_video
import tensorflow as tf

YOLO_PATH = "YOLOv4"

WEIGHTS_YOLO4_TINY = "yolov4-tiny.weights"

CFG_YOLO4_TINY = "yolov4-tiny.cfg"

CHECKPOINT_PATH = "batch_16_epochs_50"

SCENE = "scene"
SELECTED_SCENE = "VREP"

CAMERA_PARAMETERS_1 = "camera_1.txt"
CAMERA_PARAMETERS_2 = "camera_2.txt"

PATH_PARAMETERS_1 = os.path.join(SCENE,SELECTED_SCENE,CAMERA_PARAMETERS_1)
PATH_PARAMETERS_2 = os.path.join(SCENE,SELECTED_SCENE,CAMERA_PARAMETERS_2)

WEIGHT = os.path.join(YOLO_PATH,WEIGHTS_YOLO4_TINY)
CONFIG = os.path.join(YOLO_PATH,CFG_YOLO4_TINY)
NAMES = "coco.names"
NO_CAMERAS = 2
DISPLAY = True
FPS = 15
dist_count = 0


COLOR_RED = [0,0,240]

timer = {}
timer["Vision_sensor1"] = time.perf_counter()
timer["Vision_sensor2"] = time.perf_counter()

def get_correct_image(image):
	# necessary in order to show and analyze the image correctly:
	# the image is converted to a 8-bit unsigned int array (0-255),
	# reshaped wrt its resolution, flipped horizontally and 
	# switched from rgb to bgr.		
	return cv2.cvtColor(
		np.resize(
			#image[2] is the image, passed as a buffer of bytes
			np.frombuffer(image[2],dtype=np.uint8),
			# image[1] contains the resolution, w/ 3 channels 
			image[1] + [3]
		)[::-1,:,:],
		cv2.COLOR_RGB2BGR
	)

def getImage(msg,vs):
	frame = get_correct_image(msg)
	vs.detection_thread.frame = frame
	clock = time.perf_counter()
	if clock - timer[f"{vs.name}"] > 1/FPS:	

		vs.output_buffer.write(cv2.resize(
			frame,
			(256,256),
			interpolation = cv2.INTER_AREA
			)
		)
		timer[f"{vs.name}"] = clock

def image_callback(vision_sensor):
	return lambda msg : getImage(msg, vision_sensor)


def read_camera_params(calibration_file_path : str): #TODO: rewrite for robustness, use best practice
	params = []
	with open(calibration_file_path,'r') as f:
		lines = f.readlines()
		for line in lines:
			if not line[0] == '#':
				for value in line.strip().split(','):
					params.append(float(value.strip()))
		position, orientation, resolution, afov  = np.array(params[:3]), np.array(params[3:6]), params[6], params[7]
	return position, orientation,resolution, afov

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
	global dist_count
	if len(array) > 0:
		assert len(array[0]) >= 2
		tmp = array.copy()
		cluster , single = [], []
		while len(tmp):
			# select seed
			seed = tmp[0]
			mask = np.linalg.norm(tmp - seed,axis=1) <= maxdiff
			if np.count_nonzero(mask) > 1: 
				print(f"{dist_count}) Distance Violated!")
				dist_count+=1
				cluster.append(tmp[mask])
			else:
				single.append(tmp[mask][0])
			tmp = tmp[~mask]
		return cluster,single
	else:
		return [],[]

def main():
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	detection_threads = []
	vision_sensors = []

	parameter_cameras = [
		read_camera_params(PATH_PARAMETERS_1), 
		read_camera_params(PATH_PARAMETERS_2)
	]

	for i in range(NO_CAMERAS):
		vs_name = f"Vision_sensor{i+1}"
		vision_sensors.append(
			VisionSensor(
				vs_name,
				ImageDetectionThread(
					WEIGHT,
					CONFIG,
					NAMES,
					DISPLAY
				),
				CameraModel(*parameter_cameras[i]),
				cv2.VideoWriter(f"{vs_name}.mp4", fourcc, FPS, (256, 256))
			)
		)
		if DISPLAY:
			cv2.namedWindow(vs_name)


	with b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApi',inactivityToleranceInSec=200) as client:		
		# Coppeliasim Setup
		client.runInSynchronousMode=False
				
		visionSensorHandles = [
			client.simxGetObjectHandle(vs.name,client.simxServiceCall())[1] 
			for vs in vision_sensors
		]

		for idx, vs in enumerate(visionSensorHandles):
			client.simxGetVisionSensorImage(
				vs,
				False,
				client.simxDefaultSubscriber(image_callback(vision_sensors[idx]))
			)

		for vs in vision_sensors:
			vs.detection_thread.start()

		client.simxStartSimulation(client.simxDefaultPublisher())
		
		cv2.namedWindow("MAP")
		map_frame = 255 * np.ones((512,512,3))
		
		
		# Mainloop
		while True:
			detections_cluster, detections_single = [], []
			frame = map_frame.copy()
			for v_s in vision_sensors:
				
				if DISPLAY:
					cv2.imshow(v_s.name, v_s.detection_thread.output)

				s_det = []
				for _,detection in v_s.detection_thread.detections.items():
					# print(label)
					ground_pos = v_s.camera_model.invert_projection(
						np.array([[detection.u], [detection.v], [1]]),
						detection.height
					)
					p = list(ground_pos[:2].flatten())

					s_det.append(p)
					
				if len(s_det) > 0:
					cluster, single = points_cluster(np.array(s_det),1.5)
					detections_cluster += cluster
					detections_single += single

			detections_cluster = np.array(detections_cluster)
			detections_single = np.array(detections_single)
			points=[]

			if len(detections_single) > 0:
				indices_single = merge_cluster(detections_single,0.9)
				for idx in indices_single:
					points.append(np.average(detections_single[idx],axis=0))

			if len(detections_cluster) > 0:
				avg_c = np.array([np.average(a,axis = 0) for a in detections_cluster])
				indices_cluster = merge_cluster(avg_c,0.9)

				for idx in indices_cluster:
					for ar in max(detections_cluster[idx],key=len): #choose the bigger cluster
						points.append(np.array(ar))
					
			
			for p in points:
				pix_pos = tuple(map(int,np.rint(np.interp(p,[0,9.35],[0,512]))))
				cv2.circle(frame,pix_pos,5,COLOR_RED,-1)

			cv2.imshow("MAP",frame)
			if cv2.waitKey(1) == 27:
				for v_s in vision_sensors:
					v_s.detection_thread.kill = True
				break
			client.simxSpinOnce()
			
			
		for vs in vision_sensors:
			vs.detection_thread.join()
			vs.output_buffer.release()

		client.simxStopSimulation(client.simxDefaultPublisher())
		

		cv2.destroyAllWindows()
	print("Realtime tracking done")
				
if __name__ == "__main__":
	main()
	print("Restoring checkpoint...")
	gen = Generator()
	checkpoint = tf.train.Checkpoint(generator = gen)
	chkp_man =  tf.train.CheckpointManager(
		checkpoint,
		CHECKPOINT_PATH,
		max_to_keep=1
	)
	checkpoint.restore(chkp_man.latest_checkpoint)
	if chkp_man.latest_checkpoint:
		print("Restored from {}".format(chkp_man.latest_checkpoint))
	else:
		print("Could not restore the checkpoint, check the path or train the network and place it in this folder!\n")

	print("Generating high FPS videos")
	for i in range(1,NO_CAMERAS+1):
		create_high_fps_video(gen,f"Vision_sensor{i}.mp4",f"Vision_sensor{i}_HIGHFPS.mp4")
		print(f"High FPS video stored in : Vision_sensor{i}_HIGHFPS.mp4")
	print()