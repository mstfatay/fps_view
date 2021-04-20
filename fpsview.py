import numpy as np
import cv2 as cv
from math import pi, sin, cos

'''
FpsView is a class to draw useful information of a dron's fps camera view.


Try this:
>>>import fpsview
# img is a numpy.ndarray representing image
>>>fv = fpsview.fpsview(img.shape)   #use that line only once 
>>>fv.fpsView(img, roll=0, hdg=1, throttle=50, speed=35,...)   #call it every frame
'''
'''
TODO:If fpsview function doesn't run fast enough (in case of an old machine), rewrite function drawPitch without using mask.
To do this, before drawing lines rotate the points around center using basic math. 
NOTE: Using masks(time complexity is the total area of image) are too slow compared to drawLine function (time complexity
is the area drawn). (Time complexities might not be exactly like that but this information is still pretty useful.)
'''

def mixWithMask(img, img2, mask):
    # result_img[A] = img2[A] if mask == 255, result_img[A] = img[A] if mask == 0 for A in pixels. Eventually returns result_img.
    #mask should have values 0 or 255 exactly. Pixels whose values aren't eighter 0 or 255, don't look great.
    #images can be bgr or gray but both images should be the same type.

    mask = cv.multiply(mask, 4)  #If we multiply mask by a number, lines look clearer.
    img_back = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))
    img_front = cv.bitwise_and(img2, img2, mask=mask)
    return cv.bitwise_or(img_back, img_front)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    return qx, qy


def drawDottedLine(img,pt1,pt2,color,thickness=1,gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    start = (dist % gap)//2
    pts= []
    for i in  np.arange(start,dist+1,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    for p in pts:
        cv.circle(img,p,thickness,color,-1)


class FpsView:
    #You can customize these values (commented values can be changed from __init__ method)
    COLOR = (0, 0, 255)     #BGR 0:255 integer
    THICKNESS = 1           #integer
    #SIZE = 400             #size of the whole thing except compass, integer
    COMPASS_SIZE = 50       #size of the compass, integer
    #TEXT_SIZE              #size of texts, double
    FONT = cv.FONT_HERSHEY_SIMPLEX
    VIEW_ANGLE_Y = pi/2     #camera's max view angle around horizontal axis in radians
                        #In order to make horizon line show horizon appropriately, change VIEW_ANGLE_Y.
    #VIEW_ANGLE_X = 0.0     #camera's max view angle around vertical axis in radians
                        #If compass' intervals are confusing try to change VIEW_ANLGE_X to have a better compass looking.
    LINE_TYPE = cv.LINE_AA  #cv.LINE_4(sligthly better performance) or cv.LINE_AA(looks good)
    ######################################

    #do not change below!

    #H = 0  #height of the image, integer, img.shape[0]
    #W = 0  #width of the image, integer, img.shape[1]
    #MIDW
    #MIDH
    #COLOR_IMG
    #mask

    def __init__(self, shape):
        self.H = shape[0]
        self.W = shape[1]
        if self.H != int(self.H) or self.W != int(self.W):
            raise ValueError("shape's values must be integers!")
        if self.H<=0 or self.W<=0:
            raise ValueError("shape values must be greater then zero!")
        self.MIDW = self.W // 2
        self.MIDH = self.H // 2
        self.VIEW_ANGLE_X = self.VIEW_ANGLE_Y * self.W / self.H #You can directly put a value for it
        self.SIZE = int(self.H * 2/3)
        self.TEXT_SIZE = self.SIZE/800
        self.COLOR_IMG = np.full((self.H, self.W, 3), self.COLOR, np.uint8)
        self.compass_dict = {0:'N', 3:'NE', 6:'E', 9:'SE', 12:'S', 15:'SW', 18:'W', 21:'NW'}
        self._clearMask()


    def _drawBorder(self, img):
        cv.rectangle(img, (self.MIDW-self.SIZE//2, self.MIDH-self.SIZE//2), (self.MIDW+self.SIZE//2, self.MIDH+self.SIZE//2), self.COLOR,self.THICKNESS)


    def _drawCros(self, img):
        #its size should be proportional to SIZE. It should be alligned according to the center of the img.
        cros_len = int(self.SIZE/20)
        cv.line(img, (self.MIDW - cros_len, self.MIDH), (self.MIDW + cros_len, self.MIDH), self.COLOR, self.THICKNESS)
        cv.line(img, (self.MIDW , self.MIDH-cros_len), (self.MIDW, self.MIDH+cros_len), self.COLOR, self.THICKNESS)


    def _drawAngleLines(self, img, angle):
        #angle is roll and in radians
        #its size should be proportional to SIZE. It should be alligned according to the center of the img.
        len_out = self.SIZE/3
        len_in = self.SIZE/4
        color = (255,0,0)
        left_of_left = (self.MIDW - int(len_out * cos(angle)), self.MIDH + int(len_out * sin(angle)))
        right_of_left = (self.MIDW - int(len_in * cos(angle)), self.MIDH + int(len_in * sin(angle)))
        right_of_right = (self.MIDW + int(len_out * cos(angle)), self.MIDH - int(len_out * sin(angle)))
        left_of_right = (self.MIDW + int(len_in * cos(angle)), self.MIDH - int(len_in * sin(angle)))
        cv.line(img, left_of_left, right_of_left, self.COLOR, 2*self.THICKNESS, self.LINE_TYPE)
        cv.line(img, right_of_right, left_of_right, self.COLOR, 2*self.THICKNESS, self.LINE_TYPE)


    def _drawCompass(self, img, hdg):
        #Old. Check drawCompass2 method.
        #hdg : heading in radians
        #its size should be alligned according to a corner of the img. Its size should be proportional to COMPASS_SIZE.
        angle = hdg
        leng = self.COMPASS_SIZE//2
        center_point = (0,0)
        compass_pos = (self.W - self.COMPASS_SIZE, self.H - self.COMPASS_SIZE)
        text_t = 3 * self.COMPASS_SIZE // 4
        text_size = self.COMPASS_SIZE / 150

        show_point = (int(-leng*sin(angle)), int(-leng*cos(angle)))

        cv.circle(img, compass_pos, self.COMPASS_SIZE-2*self.THICKNESS, self.COLOR, self.THICKNESS, self.LINE_TYPE)
        cv.putText(img, 'N', (compass_pos[0], compass_pos[1] - text_t), self.FONT, text_size, self.COLOR, self.THICKNESS, self.LINE_TYPE)
        cv.putText(img, 'S', (compass_pos[0], compass_pos[1] + text_t), self.FONT, text_size, self.COLOR, self.THICKNESS, self.LINE_TYPE)
        cv.putText(img, 'E', (compass_pos[0] + text_t, compass_pos[1]), self.FONT, text_size, self.COLOR, self.THICKNESS, self.LINE_TYPE)
        cv.putText(img, 'W', (compass_pos[0] - text_t, compass_pos[1]), self.FONT, text_size, self.COLOR, self.THICKNESS, self.LINE_TYPE)
        cv.line(img, tuple(map(sum, zip(center_point, compass_pos))), tuple(map(sum, zip(show_point, compass_pos))), self.COLOR, self.THICKNESS,self.LINE_TYPE)


    def _drawThrottle(self, img, throttle):
        #throttle is between 0 and 100 and a float
        # its size should be proportional to SIZE. It should be alligned according to the center of the img.
        distance = 2*self.SIZE//5
        half_height = self.SIZE//3
        heigth_of_bar = int(throttle*2*half_height)
        cv.rectangle(img, (self.MIDW-distance-self.SIZE//50, self.MIDH+half_height), (self.MIDW-distance+self.SIZE//50, self.MIDH-half_height),
                     self.COLOR, self.THICKNESS,self.LINE_TYPE)
        cv.rectangle(img, (self.MIDW-distance-self.SIZE//100, self.MIDH+half_height), (self.MIDW-distance+self.SIZE//100, self.MIDH+half_height-heigth_of_bar),
                     self.COLOR, cv.FILLED)
        cv.putText(img, 'THR', (self.MIDW-distance-2*self.SIZE//50, self.MIDH-half_height-self.SIZE//100),
                   self.FONT, self.TEXT_SIZE, self.COLOR, self.THICKNESS, self.LINE_TYPE)
        cv.putText(img, str(int(throttle*100))+'%', (self.MIDW - distance - 6*self.SIZE//50, self.MIDH),
                   self.FONT, self.TEXT_SIZE,self.COLOR, self.THICKNESS, self.LINE_TYPE)


    def _drawSpeed(self, img, speed):
        #speed in meter/second float
        #remind: distance and half_height should be the same with drawTrottle function's values.
        speed_text = 'SPD'
        speed_text2 = str(round(speed)) + 'm/s'
        distance = 2 * self.SIZE // 5
        half_height = self.SIZE // 3
        cv.putText(img, speed_text, (self.MIDW - distance - int(30*self.TEXT_SIZE), self.MIDH + half_height + int(60*self.TEXT_SIZE)),
                   self.FONT, self.TEXT_SIZE, self.COLOR, self.THICKNESS, self.LINE_TYPE)
        cv.putText(img, speed_text2, (self.MIDW - distance - int(50*self.TEXT_SIZE), self.MIDH + half_height + int(90*self.TEXT_SIZE)),
                   self.FONT, self.TEXT_SIZE, self.COLOR, self.THICKNESS, self.LINE_TYPE)

    def _drawClimb(self, img, climb):
        #climb in meter/second float
        distance = 2 * self.SIZE // 5
        half_height = self.SIZE // 3
        heigth_of_bar = int(climb / 10 * half_height)
        cv.rectangle(img, (self.MIDW + distance + self.SIZE // 50, self.MIDH + half_height),
                     (self.MIDW + distance - self.SIZE // 50, self.MIDH - half_height), self.COLOR, self.THICKNESS, self.LINE_TYPE)

        if -10<climb and climb<10:
            cv.rectangle(img, (self.MIDW + distance + self.SIZE // 100, self.MIDH),
                         (self.MIDW + distance - self.SIZE // 100, self.MIDH - heigth_of_bar), self.COLOR,
                         cv.FILLED)
        else:
            sign = 1 if climb>0 else -1
            cv.rectangle(img, (self.MIDW + distance + self.SIZE // 50, self.MIDH),
                         (self.MIDW + distance - self.SIZE // 50, self.MIDH - half_height * sign), self.COLOR, cv.FILLED)

        cv.putText(img, 'CLB', (self.MIDW + distance - 2 * self.SIZE // 50, self.MIDH - half_height - self.SIZE // 100),
                   self.FONT, self.TEXT_SIZE, self.COLOR, self.THICKNESS, self.LINE_TYPE)
        cv.putText(img, str(round(climb)) + 'm/s', (self.MIDW + distance + self.SIZE // 50, self.MIDH), self.FONT,
                   self.TEXT_SIZE, self.COLOR, self.THICKNESS,self.LINE_TYPE)


    def _drawAltitude(self, img, altitude):
        #altitude in meters, float
        altitude_text = 'ALT'
        altitude_text2 = str(round(altitude)) + 'm'
        distance = 2 * self.SIZE // 5
        half_height = self.SIZE // 3
        cv.putText(img, altitude_text, (
        self.MIDW + distance - int(30 * self.TEXT_SIZE), self.MIDH + half_height + int(60 * self.TEXT_SIZE)), self.FONT,
                   self.TEXT_SIZE, self.COLOR, self.THICKNESS, self.LINE_TYPE)
        cv.putText(img, altitude_text2, (
        self.MIDW + distance - int(30 * self.TEXT_SIZE), self.MIDH + half_height + int(90 * self.TEXT_SIZE)), self.FONT,
                   self.TEXT_SIZE, self.COLOR, self.THICKNESS, self.LINE_TYPE)


    def _drawPitch(self, img, pitch, is_mask=False):
        #pitch and roll are in radiands, float
        #if is_mask == True img should be GRAY, otherwise BGR.

        if is_mask:
            color = 255
        else:
            color = self.COLOR

        gap = self.H/self.VIEW_ANGLE_Y * pi/18
        zero_line = int(self.H/self.VIEW_ANGLE_Y * pitch)

        uppest = (zero_line + self.SIZE//3) // int(gap)
        lowest = (self.SIZE//3 - zero_line) // int(gap)
        for i in range(-uppest, lowest+1):
            h = int(i* gap)
            if i == 0:
                pass
                #This line is drawn by drawHorizon method
                #cv.line(img, (self.MIDW - self.W//9, self.MIDH + zero_line + h), (self.MIDW + self.W//9, self.MIDH +  zero_line +h), color, self.THICKNESS)
            elif i%18 == 0:
                # Dotted line at 180 degrees.
                drawDottedLine(img, (self.MIDW - self.SIZE // 4, self.MIDH + zero_line + h),
                               (self.MIDW + self.SIZE // 4, self.MIDH + zero_line + h),color, self.THICKNESS, 7)
            else:
                text_value = -i * 10
                text_value = (text_value+180)%360 - 180
                if text_value==-180:
                    text_value = 180
                cv.putText(img, str(text_value), (self.MIDW + self.SIZE // 9, self.MIDH + zero_line + h), self.FONT,
                           self.TEXT_SIZE * 0.75, color, self.THICKNESS)
                cv.line(img, (self.MIDW - self.SIZE//10, self.MIDH + zero_line + h), (self.MIDW + self.SIZE//10, self.MIDH +  zero_line +h),
                        color, self.THICKNESS)


    def _drawHorizon(self, img, pitch, roll):
        color = (0,255,0)
        gap = self.H / self.VIEW_ANGLE_Y * pi / 18
        zero_line = int(self.H / self.VIEW_ANGLE_Y * pitch)

        uppest = (zero_line + self.SIZE // 3) // int(gap)
        lowest = (self.SIZE // 3 - zero_line) // int(gap)
        if -lowest<=0 and 0<=uppest:
            point1 = (self.MIDW - self.SIZE // 4, self.MIDH + zero_line)
            point2 = (self.MIDW + self.SIZE // 4, self.MIDH + zero_line)
            point1 =  tuple(map(int, rotate((self.MIDW, self.MIDH), point1, -roll)))
            point2 = tuple(map(int, rotate((self.MIDW, self.MIDH), point2, -roll)))
            cv.line(img, point1, point2, color, self.THICKNESS, self.LINE_TYPE)


    def _drawCompass2(self, img, hdg):
        # Draws a cool compass on top of the view
        #hdg in degrees
        points = [[self.MIDW, self.MIDH - self.SIZE // 2 + self.SIZE // 20],
                  [self.MIDW - self.SIZE // 40, self.MIDH - self.SIZE // 2 + self.SIZE // 10],
                  [self.MIDW + self.SIZE // 40, self.MIDH - self.SIZE // 2 + self.SIZE // 10]]
        cv.fillConvexPoly(img, np.array(points, np.int32).reshape((-1,1,2)), self.COLOR) #triangle

        hdg = hdg/180 * pi  #hdg now in radians and float
        gap = self.W/self.VIEW_ANGLE_X * pi/12 / 3
        N_line = int(-self.W/self.VIEW_ANGLE_X * hdg)

        leftest = (self.SIZE//2 + N_line) // int(gap)
        rightest = (self.SIZE//2 - N_line) // int(gap)

        for i in range(-leftest, rightest+1):
            h = int(i * gap)

            # in order to keep values between 0 and 360
            if i>=24 *3:
                i -= 24 *3
            elif i<0:
                i += 24 *3

            if i%3==0:
                #draw long line
                cv.line(img, (self.MIDW + N_line + h, self.MIDH - self.SIZE // 2),
                        (self.MIDW + N_line + h, self.MIDH - self.SIZE // 2 + self.SIZE // 20), self.COLOR,
                        self.THICKNESS, self.LINE_TYPE)
            else:
                #draw short line
                cv.line(img, (self.MIDW + N_line + h, self.MIDH - self.SIZE // 2),
                        (self.MIDW + N_line + h, self.MIDH - self.SIZE // 2 + self.SIZE // 40), self.COLOR,
                        self.THICKNESS, self.LINE_TYPE)
            if i%9==0:
                #put N,NE,E,... texts
                cv.putText(img, self.compass_dict[i/3], (self.MIDW + N_line + h, self.MIDH - self.SIZE // 2), self.FONT,
                           self.TEXT_SIZE, self.COLOR, self.THICKNESS, self.LINE_TYPE)
            elif i%3==0:
                #put 15,30,45,105,... texts
                cv.putText(img, str(i//3*15), (self.MIDW + N_line + h, self.MIDH - self.SIZE//2),
                           self.FONT, self.TEXT_SIZE, self.COLOR, self.THICKNESS, self.LINE_TYPE)

    def _clearMask(self):
        self.mask = np.zeros((self.H, self.W), np.uint8)



    def fpsView(self, img, speed=0.0, hdg=0, throttle=0.0, altitude=0.0, climb=0.0, roll=0.0, pitch=0.0):
        '''The given image isn't changed. This function returns the result image.
        speed  [0:)         float   m/s
        hdg [0:360)         int     deg
        throttle [0.0, 1.0] float
        altitude            float   m
        climb               float   m/s
        roll                float   rad
        pitch [-pi,pi]      float   rad
        '''
        if type(img) != np.ndarray:
            raise ValueError("img must be a np.ndarray!")
        if img.shape[0]!=self.H or img.shape[1]!=self.W or img.shape[2]!=3:
            raise ValueError("Shape of img is wrong!")
        result_img = np.copy(img)
        self._clearMask()
        #self._drawBorder(img)   #a border might be useful while customizing
        self._drawAngleLines(result_img, roll)
        #self._drawCompass(img, hdg)  #old compass
        self._drawThrottle(result_img, throttle)
        self._drawSpeed(result_img, speed)
        self._drawClimb(result_img,climb)
        self._drawHorizon(result_img, pitch, roll)
        self._drawAltitude(result_img,altitude)
        self._drawPitch(self.mask, pitch, is_mask=True)
        self._drawCompass2(result_img, hdg)
        self._drawCros(result_img)

        #rotate pitch mask and mix with img
        m = cv.getRotationMatrix2D((self.MIDW, self.MIDH), roll/pi*180, 1)
        self.mask = cv.warpAffine(self.mask, m, (self.W, self.H))
        result_img = mixWithMask(result_img, self.COLOR_IMG, self.mask)

        return result_img


def nothing(x):
    pass

if __name__ == '__main__':
    #This part is to test fpsView function using a picture.
    img = None
    if img is None:
        img = np.zeros((480,720,3), dtype=np.uint8)
    fpsview = FpsView(img.shape)

    winname = 'fpsview'
    cv.namedWindow(winname, cv.WINDOW_AUTOSIZE)
    roll = 180
    hdg = 180
    throttle = 57
    speed = 35
    climb = 10
    altitude = 0
    pitch = 180
    cv.createTrackbar('roll', winname, roll, 360, nothing)
    cv.createTrackbar('hdg', winname, hdg, 360, nothing)
    #cv.createTrackbar('thr', winname, throttle, 100, nothing)
    #cv.createTrackbar('spd', winname, speed, 150, nothing)
    cv.createTrackbar('clmb', winname, climb, 20, nothing)
    #cv.createTrackbar('alt', winname, altitude, 100, nothing)
    cv.createTrackbar('pit', winname, pitch, 360, nothing)

    ch = 0
    while (ch != 27):
        roll = cv.getTrackbarPos('roll', winname)
        hdg = cv.getTrackbarPos('hdg', winname)
        #throttle = cv.getTrackbarPos('thr', winname)
        #speed = cv.getTrackbarPos('spd', winname)
        climb = cv.getTrackbarPos('clmb', winname)
        #altitude = cv.getTrackbarPos('alt', winname)
        pitch = cv.getTrackbarPos('pit', winname)

        #disp_img = np.copy(img)
        disp_img = fpsview.fpsView(img,
                        roll=(roll-180)*pi/180,
                        hdg=hdg-180,
                        throttle=throttle/100,
                        speed=speed,
                        climb=climb-10,
                        altitude=altitude,
                        pitch=(pitch-180)/180*pi)
        cv.imshow(winname, disp_img)
        ch = cv.waitKey(1) & 0xFF

    cv.destroyAllWindows()