from dataclasses import dataclass
from detection_thread import ImageDetectionThread
from CameraModel import CameraModel

@dataclass
class VisionSensor:
	name : 				str
	detection_thread : 	ImageDetectionThread
	camera_model : 		CameraModel