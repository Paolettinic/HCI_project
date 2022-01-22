import b0RemoteApi

read_heights = []
person_heights = []

def compute_distance(distance):
    print(2-distance)
def read_proximity_sensor(msg):
    global read_heights
    global person_heights
    if msg:
        if msg[1] > 0:
            read_heights.append(msg[3][2])
        else:
            if len(read_heights) > 0:
                person_heights.append(2.0 - min(read_heights))
                read_heights = []



def main():
    global stop
    proximity_sensor_name = "door_top_sensor"
    proximity_sensor_height = 2.0
    with b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApi',inactivityToleranceInSec=200) as client:
        laser_scanner_handler = client.simxGetObjectHandle(proximity_sensor_name,client.simxServiceCall())[1]
        client.simxStartSimulation(client.simxDefaultPublisher())
        client.simxReadProximitySensor(
                laser_scanner_handler,
                client.simxDefaultSubscriber(read_proximity_sensor)
        )
        try:
            old_len = len(person_heights)
            while True:
                client.simxSpinOnce()  
                new_len = len(person_heights)
                if old_len is not new_len:
                    print("Person height: ",person_heights)
                    old_len = new_len
        except KeyboardInterrupt:
            client.simxStopSimulation(client.simxDefaultPublisher())
            exit(0)
        
if __name__ == "__main__":
    main()