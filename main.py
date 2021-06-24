from pyrep import VRep
from CameraModel import Camera
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    #img = plt.imread('Lenna_(test_image).png')
            Width = img.shape[1]
            Height = img.shape[0]
            net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
            #net.setInput(cv2.dnn.blobFromImage(img, 1, camera1.resolution, (0,0,0), True, crop=False))
            net.setInput(cv2.dnn.blobFromImage(img, 1, (Width,Height), (0,0,0), True, crop=False))
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

    


            for out in outs:
                for detection in out:
                    # print("Detected Something")
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.1:
                        print(f"Detected with confidence {confidence}")
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)

                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)

                        x = center_x - w / 2
                        y = center_y - h / 2

                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)


            for i in indices:
                i = i[0]
                box = boxes[i]
                if class_ids[i]==0:
                    label = str("Persona") #ID della label persona 
                    cv2.rectangle(img, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (255, 0, 0), 2)
                    cv2.putText(img, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            cv2.imshow('image',img)
            key = cv2.waitKey(1)
            if key == 27:
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()