from unity_utils.unity_utils import Unity
import cv2
import time

unity_api = Unity(11000)
unity_api.connect()

while True:
    data = unity_api.set_speed_angle(150, 25)
    print(data)
    start_time = time.time()
    left_image, right_image = unity_api.get_images()
    print("time: ", 1/(time.time() - start_time))
    unity_api.show_images(left_image, right_image)