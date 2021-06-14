from pyrep import VRep
from CameraModel import Camera
import cv2

def main():
    with VRep.connect("127.0.0.1",19997) as api:
        camera1 = Camera(
                api.sensor.vision("Vision_sensor"),
                [-4.85,-4.825,2.205],
                [-123.40,39.06,-157.09],
                (256,256)
            )
        while True:
            img = camera1.getCorrectRawImage()
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            cv2.imshow('image',img)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()