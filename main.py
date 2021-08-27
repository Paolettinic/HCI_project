from pyrep import VRep
from CameraModel import Camera
import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading
from dataclasses import dataclass

def load_yolo():
	net = []
	net.append(cv2.dnn.readNet("yolov3.weights", "yolov3.cfg"))
	net.append(cv2.dnn.readNet("yolov3.weights", "yolov3.cfg"))
	net.append(cv2.dnn.readNet("yolov3.weights", "yolov3.cfg"))
	net.append(cv2.dnn.readNet("yolov3.weights", "yolov3.cfg"))
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net[0].getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net[0].getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net , classes, colors, output_layers

def detect_objects(img, net, outputLayers, height, width):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(height, width), mean=(0, 0, 0), swapRB=True, crop=False)

	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			# print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
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
	return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img, camera_name): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imshow(camera_name, img)
	
	cv2.waitKey(1)

		
def find_boxes(image, net, output_layers,classes,colors,camera_name):
	img = cv2.resize(image,None,fx=1,fy=1) #needed for cv2 img
	height, width, channels = img.shape
	blob, outputs = detect_objects(img,net,output_layers,height,width)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	print(f"CAMERA: {camera_name}")
	for i in range(len(boxes)):
		print(f"label: {classes[class_ids[i]]}")
		print(f"box: {boxes[i]}")
	# draw_labels(boxes, confs, colors, class_ids, classes, img, camera_name)

def main():
	

	with VRep.connect("127.0.0.1",19997) as api:
		net, classes, colors, output_layers = load_yolo()

		camera_parameters = [ 
			{
				"name" : "Vision_sensor",
				"handler" : api.sensor.vision("Vision_sensor"),
				"position" : [-4.85, -4.825, 2.205],
				"orientation" : [-123.40, 39.86, -157.09],
				"resolution" : (416, 416)
			},
			{
				"name" : "Vision_sensor0",
				"handler" : api.sensor.vision("Vision_sensor0"),
				"position" : [4.8, 4.75, 2.205],
				"orientation" : [-123.40, -39.86, 22.91],
				"resolution" : (416, 416)
			},
			{
				"name" : "Vision_sensor1",
				"handler" : api.sensor.vision("Vision_sensor1"),
				"position" : [4.8, -4.825, 2.205],
				"orientation" : [-123.40, -39.86, 157.09],
				"resolution" : (416, 416)
			},
			{
				"name" : "Vision_sensor2",
				"handler" : api.sensor.vision("Vision_sensor2"),
				"position" : [-4.85, 4.75, 2.205],
				"orientation" : [123.40, 39.86, -22.91],
				"resolution" : (416, 416)
			}
			
		]

		cameras = []
		for c_p in camera_parameters:
			cameras.append(Camera(**c_p))
		
		while True:
			threads = []
			for i, c in enumerate(cameras):
				img = c.getCorrectRawImage()
				threads.append(threading.Thread(
					target=find_boxes,
					args=(img, net[i], output_layers,classes,colors,c.name)
				))
			for thread in threads:
				thread.start()

			#immagini qui

			for thread in threads:
				thread.join()
		 
if __name__ == "__main__":
	main()


#draw_labels(boxes, confs, self.colors, class_ids, self.classes, img, self.camera_name)