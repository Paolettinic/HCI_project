
import b0RemoteApi
import cv2
import numpy as np
import time

NO_CAMERAS = 1
DISPLAY = True

dist_count = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"videoout1.mp4", fourcc, 15.0, (256, 256))

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

def getImage(msg):
	global out
	frame = cv2.resize(get_correct_image(msg), (256,256), interpolation = cv2.INTER_AREA)
	out.write(frame)



def main():
	global out
	vision_sensors = []
	t = time.perf_counter()
	with b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApi',inactivityToleranceInSec=200) as client:		
		# Coppeliasim Setup
		client.runInSynchronousMode=False
		for i in range(NO_CAMERAS):
			vision_sensors.append(f"Vision_sensor{i+1}")

		visionSensorHandles = [
			client.simxGetObjectHandle(vs,client.simxServiceCall())[1] 
			for vs in vision_sensors
		]
		

		for vs in visionSensorHandles:
			client.simxGetVisionSensorImage(
				vs,
				False,
				client.simxDefaultSubscriber(getImage)
			)


		client.simxStartSimulation(client.simxDefaultPublisher())

		# Mainloop
		while time.perf_counter() - t < 10:
			time.sleep(1/15)
			client.simxSpinOnce()
		
			
		out.release()

		client.simxStopSimulation(client.simxDefaultPublisher())
		
	print("done")
				
if __name__ == "__main__":
	main()
	print("main done")