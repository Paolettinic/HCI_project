from abc import abstractclassmethod
import threading
import cv2
import numpy as np
import time
import os

YOLO_PATH = "YOLOv3"
WEIGHTS_YOLO320 = "yolov3-320.weights"
WEIGHTS_YOLO_TINY = "yolov3-tiny.weights"
WEIGHTS_YOLO = "yolov3.weights"

CFG_YOLO320 = "yolov3-320.cfg"
CFG_YOLO_TINY = "yolov3-tiny.cfg"
CFG_YOLO = "yolov3.cfg"
SCENE_PATH = "scene"
VIDEO_PATH = "VIDEO"

WEIGHT = os.path.join(YOLO_PATH,WEIGHTS_YOLO320)
CONFIG = os.path.join(YOLO_PATH,CFG_YOLO320)
NAMES = "coco.names"
NO_CAMERAS = 1
FPS = 15
frame_buf = {f"camera{i}" : np.zeros((480,640,3)) for i in range(1,NO_CAMERAS + 1)}
output_buf = {f"camera{i}" : np.zeros((480,640,3)) for i in range(1,NO_CAMERAS + 1)}


kill = False


class Camera:
	@abstractclassmethod
	def get_image(self):
		pass


class CameraVideo(Camera,threading.Thread):
	def __init__(self, camera_name, video_path):
		threading.Thread.__init__(self)
		self.camera_name = camera_name
		self.video = cv2.VideoCapture(os.path.join(SCENE_PATH,VIDEO_PATH,video_path))
		self.sleep_time = 1 / (self.video.get(cv2.CAP_PROP_FPS) - 5) 

	def run(self):
		global frame_buf
		global kill 

		grabbed, frame = self.video.read()
		while grabbed and not kill:
			time.sleep(self.sleep_time)
			frame_buf[self.camera_name] = frame
			grabbed, frame = self.video.read()

		self.video.release()

class DetectionThread(threading.Thread):
	def __init__(self,weights_file,cfg_file,names_file,camera_name):
		threading.Thread.__init__(self)
		self.net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

		self.classes = []
		with open(names_file, "r") as f:
			self.classes = [line.strip() for line in f.readlines()]
		layers_names = self.net.getLayerNames()
		self.output_layers = [layers_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()]
		self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
		self.camera_name = camera_name
		cv2.namedWindow(self.camera_name)

	def run(self):
		global kill
		global frame_buf
		global output_buf
		
		while not kill:
			frame = frame_buf[self.camera_name]
			if frame is not None:
				height, width, channels = frame.shape
				blob = cv2.dnn.blobFromImage(np.float32(frame), scalefactor=1/255, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
				self.net.setInput(blob)
				outputs = self.net.forward(self.output_layers)
				boxes = []
				confs = []
				class_ids = []
				for output in outputs:
					for detect in output:
						scores = detect[5:]
						class_id = np.argmax(scores)
						conf = scores[class_id]
						if class_id == 0:
							if conf > 0.5:
								center_x = int(detect[0] * width)
								center_y = int(detect[1] * height)
								w = int(detect[2] * width)
								h = int(detect[3] * height)
								x = int(center_x - w/2)
								y = int(center_y - h / 2)
								boxes.append([x, y, w, h])
								confs.append(float(conf))
								class_ids.append(class_id)
				indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.2)
				font = cv2.FONT_HERSHEY_PLAIN
				if len(indexes)>0:
					for i in indexes.flatten():
						x, y, w, h = boxes[i]
						label = str(self.classes[class_ids[i]])
						color = self.colors[i]
						cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
						cv2.putText(frame, f"{label}: {confs[i] :.2f}", (x, y - 5), font, 1, color, 1)
					
				output_buf[self.camera_name] = frame	
				
		

def main():
	global kill
	camera_names = []
	camera_threads = []
	detection_threads = []

	for i in range(1,NO_CAMERAS + 1):
		camera_names.append(f"camera{i}")
		camera_threads.append(CameraVideo(f"camera{i}",f"videoout{i}.mp4"))

		detection_threads.append(
			DetectionThread(
				WEIGHT,
				CONFIG,
				NAMES,
				f"camera{i}"
			)
		)
	for thread in detection_threads:
		thread.start()

	for camera_thread in camera_threads:
		camera_thread.start()
		

	while True:
		for camera in camera_names:
			cv2.imshow(camera,output_buf[camera])
		
		if cv2.waitKey(1) == 27:
			kill = True
			break

	for camera_thread in camera_threads:
		camera_thread.join()

	for thread in detection_threads:
		thread.join()

	cv2.destroyAllWindows()
		
			
			
		

			
		

if __name__ == "__main__":
	main()


