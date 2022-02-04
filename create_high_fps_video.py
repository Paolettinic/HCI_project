from generator_model import Generator
import cv2
import numpy as np
import tensorflow as tf

def create_high_fps_video(gen, video_path, output_path):

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output_path, fourcc, 15+14, (256, 256))
	video = cv2.VideoCapture(video_path)

	grabbed, prev_frame = video.read()

	out.write(prev_frame)

	grabbed, frame = video.read()
	while grabbed:
		prev_frame_batch = np.expand_dims(prev_frame,0)/127.5 - 1
		frame_batch = np.expand_dims(frame,0)/127.5 - 1
		new_frame = (np.squeeze(gen([prev_frame_batch, frame_batch]),0)+1)*127.5
		new_frame = new_frame.astype(np.uint8)
		out.write(new_frame)
		out.write(frame)
		prev_frame = frame
		grabbed, frame = video.read()


	out.release()