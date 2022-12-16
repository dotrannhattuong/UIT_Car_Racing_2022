import time
import numpy as np
import cv2
from UITCar import UITCar

Car = UITCar()

pre_t = time.time()
error_arr = np.zeros(5)

def remove_small_contours(image):
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
    return image_remove
    
#   Hàm dự đoán làn đường dựa vào ảnh từ camera
def road_lines(image, session, inputname):
	# Crop ảnh lại, lấy phần ảnh có làn đườngs
	image = image[200:, :, :]
	small_img = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
	small_img = small_img/255
	small_img = np.array(small_img, dtype=np.float32)
	small_img = small_img[None, :, :, :]
	prediction = session.run(None, {inputname: small_img})
	prediction = np.squeeze(prediction)
	prediction = np.where(prediction < 0.5, 0, 255)
	# prediction = prediction.reshape(small_img.shape[0], small_img.shape[1])
	prediction = prediction.astype(np.uint8)

	return prediction

#   Hàm dự đoán làn đường dựa vào ảnh từ camera và trả về góc lái, center
# def road_lines(image, session,inputname):
# 	# Crop ảnh lại, lấy phần ảnh có làn đường
# 	image=image[200:,:,:]
# 	small_img = cv2.resize(image, (160, 80))
# 	 # cv2.imshow("crop",small_img)
# 	small_img = np.array(small_img,dtype=np.float32)
# 	small_img = small_img[None, :, :, :]
# 	prediction = session.run(None,{inputname:small_img})
# 	prediction=np.squeeze(prediction)
# 	# prediction = prediction*255
# 	prediction = np.where(prediction < 0.5, 0, 255)
# 	prediction = prediction.reshape(80, 160)
# 	 # print(prediction.shape)
# 	prediction = prediction.astype(np.uint8)

# 	return prediction
    
class Controller:
    def __init__(self, kp, kd):
        self.__kp = kp
        self.__kd = kd
        self.__LANEWIGHT = 70            # Độ rộng đường (pixel)
        self.width = 70

        self.mode = 0
        self.__turn_mode = 'None'

        self.__pre_error = 0
        self.__offset = 4
        Car.setMotorMode(0)

    def findingLane(self, frame, height):
        arr_normal = []
        lineRow = frame[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr_normal.append(x)
        if not arr_normal:
            # arr_normal = [frame.shape[1] * 1 // 3, frame.shape[1] * 2 // 3]
            return 0

        self.minLane = min(arr_normal)
        self.maxLane = max(arr_normal)
        
        center = int((self.minLane + self.maxLane) / 2)
        
        #### Safe Mode ####
        # self.width=self.maxLane-self.minLane
        # print(self.width)
        # # print(self.minLane, self.maxLane)
        # if self.__turn_mode == 'straight':
        #     if 0 <= self.minLane <= 40 and 80 <= self.maxLane <= 135 and self.width < self.__LANEWIGHT:
        #         center = self.maxLane - self.__LANEWIGHT//2
        #         print('Change Max', self.minLane, self.maxLane)
        #     elif 35 <= self.minLane <= 80 and self.maxLane >= 129 and self.width < self.__LANEWIGHT:
        #         center = self.minLane + self.__LANEWIGHT//2
        #         print('Change Min', self.minLane, self.maxLane)
        #     elif self.width > 110:
        #         self.__turn_mode = 'None'

        #### Cua sớm ####
        # if self.__turn_mode == 'None':
        if (0 < self.width < self.__LANEWIGHT):
            if (center < int(frame.shape[1]/2)):
                center -= self.__LANEWIGHT - self.width
            else :
                center += self.__LANEWIGHT - self.width

        #### Error ####
        error = frame.shape[1]//2 - center
        self.__pre_error = error

        return error

    def __PID(self, error):
        global pre_t
        # global error_arr
        error_arr[1:] = error_arr[0:-1]
        error_arr[0] = error
        P = error*self.__kp
        delta_t = time.time() - pre_t
        pre_t = time.time()
        D = (error-error_arr[1])/delta_t*self.__kd
        angle = P + D
        
        if angle + self.__offset >= 50 + self.__offset:
            angle = 50 + self.__offset
        elif angle + self.__offset <= -50 + self.__offset:
            angle = -50 + self.__offset

        return int(angle)

    def __show(self, signal, area):
        Car.OLED_Print(f"{signal}     {area}", line = 0, left = 0)
        Car.OLED_Print(f"{self.minLane}", line = 1, left = 0)
        Car.OLED_Print(f"{self.maxLane}", line = 2, left = 0)
        Car.OLED_Print(f"{self.width}", line = 3, left = 0)
    
    def __timer_intersection(self, time_turn, angle_turn, velo):
        Car.setSpeed_rad(velo)

        # if self.__turn_mode == 'straight':
        #     print(self.minLane)
        #     width_turn = 100
        #     if self.minLane <= 30: 
        #         print('okkkkkkkkkkkk')
        #         angle_turn = -5
        #     elif self.maxLane >= 139:
        #         angle_turn = 6
            
        # else:
        #     width_turn = 120

        if self.width >= 110: # 110
            print('Timerrrrrrrrrrrrrrrrrr')
            Car.setAngle(angle_turn) #### Set Angle

            start_time = time.time()
            while(time.time() - start_time < time_turn):
                ### Safe Mode ###
                if Car.button == 3:
                    Car.setAngle(angle_turn)
                    Car.setSpeed_rad(velo)
                elif Car.button == 2:
                    Car.setAngle(0)
                    Car.setSpeed_rad(0)

            self.__turn_mode = 'None'
        else:
            Car.setAngle(self.__sendBack_angle + self.__offset)
            Car.setSpeed_rad(self.__sendBack_speed)


    def __start_stop(self, sendBack_angle, sendBack_speed):
        if Car.button == 3:
            if self.__turn_mode == 'None':
                Car.setAngle(sendBack_angle + self.__offset)
                Car.setSpeed_rad(sendBack_speed)

            else:
                if self.__turn_mode == 'left':
                    self.__turnleft()
                elif self.__turn_mode == 'right':
                    self.__turnright()
                elif self.__turn_mode == 'straight':
                    self.__straight()

        elif Car.button == 2:
            Car.setAngle(0)
            Car.setSpeed_rad(0)

    def __turnright(self):
        print('Rightttttttttttttttttttttttt')
        self.__timer_intersection(time_turn=1, velo=25, angle_turn=-45) # -45

    def __turnleft(self):
        print('Leftttttttttttttttttttttt')
        self.__timer_intersection(time_turn=1, velo=25, angle_turn=50)
    
    def __straight(self):
        print('Straightttttttttttttttttt')
        self.__timer_intersection(time_turn=0.2, velo=28, angle_turn=self.__offset)

    def __linear(self, error):
        return int(-0.15*abs(error) + 25)

    def __call__(self, frame, sendBack_speed, height, signal, area):
        self.__frame = frame
        ####### Angle Processing #######
        self.error = self.findingLane(frame, height)
        self.__sendBack_angle = self.__PID(self.error)

        ####### Speed Processing #######
        self.__sendBack_speed = self.__linear(self.error)

        ####### Show Info #######
        # self.__show(signal, area)

        #### Traffic Sign Processing ####
        # if signal != 'thang' and signal != 'None' and 1000 <= area <= 2000:
        #     print(signal, area)
        #     sendBack_speed = 15
        
        if 1500 <= area <= 4000:
            print(signal, area)
            if signal == 'camtrai':
                self.__turn_mode = 'right'
            elif signal == 'camphai':
                self.__turn_mode = 'left'
            elif signal == 'phai':
                self.__turn_mode = 'right'
            elif signal == 'trai':
                self.__turn_mode = 'left'
            elif signal == 'thang':
                self.__turn_mode = 'straight'

        ####### Start-Stop #######
        # print(sendBack_speed)
        self.__start_stop(self.__sendBack_angle, self.__sendBack_speed)

        return self.__sendBack_angle, self.__sendBack_speed