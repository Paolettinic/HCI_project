import threading
import numpy as np
import cv2
from sort import Sort
from queue import Queue





class Detection:
	def __init__(self, x1, y1, x2, y2) -> None:
		self.u = 0
		self.v = 0
		self.update_bbox(x1,y1,x2,y2)

		self.height = 1.67 # use an approximate height to compute an approximate position

		self.deleted = False
		self.from_last_update = 0

	def update_bbox(self, x1, y1, x2, y2):
		self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
		self.u = int((x1+x2)/2)
		self.v = y1
		self.from_last_update = 0

class ImageDetectionThread(threading.Thread):

	def __init__(self,weights_file:str,cfg_file:str,names_file:str,display = False):
		threading.Thread.__init__(self)
		print("DETECTION_THREAD initialized")
		self.sensor_position = np.array([1.2,0])
		self.classes = []
		with open(names_file, "r") as f:
			self.classes = [line.strip() for line in f.readlines()]
		self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

		self.net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		self.frame = np.zeros((512,512,3))
		self.output = np.zeros((512,512,3))
		layers_names = self.net.getLayerNames()
		self.output_layers = [layers_names[i-1] for i in self.net.getUnconnectedOutLayers().flatten()]
		self.kill = False
		self.sort = Sort()
		self.detections = {}
		self.display = display
		# if display:
		# 	cv2.namedWindow(self.camera_model.name) #Create opencv window to display the image 

	def run(self):
		print("Camera thread called")
		while not self.kill:
			self.detect()
		print("Detection thread closed")

	def detect(self):
		color = [225,40,10]
		frame = self.frame
		if frame is not None:
			height, width, channels = frame.shape

			# Convert input frame to blob
			blob = cv2.dnn.blobFromImage(
				np.float32(frame), 
				scalefactor=1/255, 
				size=(416, 416), 
				mean=(0, 0, 0),
				swapRB=True,
				crop=False
			)
			# YOLO Inference
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
					if class_id == 0 and conf > 0.4:
						center_x = int(detect[0] * width)
						center_y = int(detect[1] * height)
						w = int(detect[2] * width)
						h = int(detect[3] * height)
						x = int(center_x - w/2)
						y = int(center_y - h / 2)
						boxes.append([x,y,x+w,y+h])
						confs.append(float(conf))
						class_ids.append(class_id)
			
			indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.4, 0.8)
			font = cv2.FONT_HERSHEY_PLAIN

			# SORT update
			if len(indexes) > 0:

				track_ids = self.sort.update(
					np.array([
						boxes[i] + [confs[i]] for i in indexes.flatten()
					])
				)
							
				update = []
				for track in track_ids:
					x1,y1,x2,y2,label = map(int,track)
					update.append(label)
					if label not in self.detections:
						self.detections[label] = Detection(x1, y1, x2, y2)

					else: 
						self.detections[label].update_bbox(x1, y1, x2, y2)	
					# OPENCV frame editing
					if self.display:
						
						cv2.rectangle(frame, (x1,y1), (x2, y2), color, 2)
						
						cv2.circle(frame, (int((x1+x2)/2),y1),2,color,-1)

						cv2.putText(frame, f"{label}", (x2, y2 - 5), font, 1, color, 1)
				
				# Life span of a detection before dying 
				
				to_delete = []

				for d in self.detections:
					if self.detections[d].from_last_update >= 3:
						to_delete.append(d)
					elif d not in update:
						self.detections[d].from_last_update += 1

				for d in to_delete:
					self.detections.pop(d)

			else:
				# Sort algorithm needs to be updated even though there are no detections
				self.sort.update()

			if self.display:
				self.output = frame

