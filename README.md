# HCI_project

## Requirements
- python > 3.8
- CoppeliaSim
- B0RemoteApi
- Opencv compiled for cuda drivers (optional)
- Numpy
- Scipy
- Filterpy
- Tensorflow V2
- YOLOv4 weights and cfg

## Installation instructions
- Install CoppeliaSim from the [official website](https://coppeliarobotics.com/)
- Follow the instructions to install B0RemoteAPI for Python: https://www.coppeliarobotics.com/helpFiles/en/b0RemoteApiClientSide.htm
- Follow the instructions on [this site](https://towardsdatascience.com/opencv-cuda-aws-ec2-no-more-tears-60af2b751c46) to download and compile OpenCV with Cuda drivers, alternatively install OpenCV from the PIP repository.

		$ sudo apt install pip3	
		$ pip3 install opencv-python 

- Install the required packages:
		
		$ pip3 install numpy
		$ pip3 install scipy
		$ pip3 install filterpy
		$ pip3 install tensorflow


## Simulation instruction
- Start CoppeliaSim and open the file located in scene/VREP/Elective.ttt
- Open a bash or a terminal and execute

		$ python3 main.py