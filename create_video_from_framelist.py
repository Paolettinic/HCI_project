import cv2
import os 

def create_synchronized_videos():
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	calibration_params = {}

	# Since images are not synchronized, here the last first frame
	# and the fist last frame are used as boundary, discarding 
	# all other images out. 
	first_frame_override = ["1221570672","828706"]
	first_frame_time = ["0","0"] 
	last_frame_time = ["9999999999","9999999999"]
	for i in range(1,5):

		with open(f"LAB/cam{i}/index.dmp","rb") as f:
			first_line_time = f.readline().decode().split()[1:]
			if first_line_time > first_frame_time:
				first_frame_time = first_line_time


			f.seek(-2, os.SEEK_END)
			while f.read(1) != b'\n':
				f.seek(-2, os.SEEK_CUR)
			last_line_time = f.readline().decode().split()[1:]
			if last_line_time < last_frame_time:
				last_frame_time = last_line_time

	
	for i in range(1,5):
		print(f"Creating video for camera {i}")
		out = cv2.VideoWriter(f"videoout{i}.mp4", fourcc, 15.0, (640, 480))
		with open(f"LAB/cam{i}/index.dmp") as f:
			lines = f.readlines()
			for line in lines:
				image_path, current_line_time = line.split(maxsplit=1)
				if first_frame_override <= current_line_time.split() <= last_frame_time:
					out.write(cv2.imread(f"LAB/cam{i}/{image_path}"))

		out.release()
		print(f"videoout{i}.mp4 created.")
		

if __name__ == "__main__":
	create_synchronized_videos()