from dataclasses import dataclass
from detection_thread import ImageDetectionThread
from CameraModel import CameraModel
from cv2 import VideoWriter
@dataclass
class VisionSensor:
	name : 				str
	detection_thread : 	ImageDetectionThread
	camera_model : 		CameraModel
	output_buffer:		VideoWriter