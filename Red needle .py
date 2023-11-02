import cv2 
import numpy as np
import matplotlib.pyplot as plt
from math import *
import pyautogui

img = cv2.imread(r'12-10-22 01-09-01.05 PM LCD Check 2222.png',cv2.IMREAD_COLOR)
print("shape",img.shape)

y=102                          #cropping an image
x=895
img= img[y:960, x:1623]
result = img.copy()
lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l,a,b = cv2.split(lab_img)     # splitting the image in l,a,b channels

ret, thresh1 = cv2.threshold(a,155,255, cv2.THRESH_BINARY)
img_normalized = cv2.normalize(thresh1, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
a5 = img_normalized.astype(np.uint8)

contours1= cv2.findContours(image=a5, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)[-2]
x,y,w,h=cv2.boundingRect(l[0])

l=[]                           # list of contours
for c in contours1:
  if (cv2.contourArea(c)>2400):
    l.append(c)
    print("countour area:",cv2.contourArea(c))
    cv2.drawContours(image=img, contours=l[0], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
# ground bounding box
x,y,w,h=cv2.boundingRect(l[0])
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
img=cv2.drawContours(img,c,0,(0,0,255), 2)
rect=cv2.minAreaRect(l[0])
box = cv2.boxPoints(rect)
box = np.intp(box)
print("box",box)
rect=cv2.minAreaRect(l[0])
print("******************rectr",rect)

img=cv2.drawContours(img,[box],0,(255,0,0), 2)

center_x,center_y=(346,488)       #centre points of the needle 

image=cv2.circle(img,( center_x,center_y), radius=4, color=(255,0,0), thickness=-1)
axis_color = (0, 255, 0)  # Green color for the axes (BGR format)

# Draw the X-axis (horizontal line)
cv2.line(img, (0, center_y), (image.shape[1], center_y), axis_color, 2)

# Draw the Y-axis (vertical line)
cv2.line(img, (center_x, 0), (center_x, image.shape[0]), axis_color, 2)

img=cv2.drawContours(img,[box],0,(255,0,0), 4)

# Function to check quadrant
def quadrant(x, y):

	if (0<x<346 and 0<y<490):
		angle=rect[2]
		print ("lies in First quadrant",angle)
		return angle

	elif (0<x<346 and 489<y <857):
		angle=rect[2]+90
		print ("lies in Second quadrant",angle)
		return angle
		
	elif (346<x<724 and 0<y<486):
		angle=rect[2]+180
		print ("lies in Third quadrant",angle)
		return angle
	
	elif (349<x<729 and 486<y<857):
		angle=rect[2]+270
		print ("lies in Fourth quadrant",angle)
		return angle

quadrant(x, y)

# function to get brightness 
def get_brightness(x, y):
    lab_pixel = lab_img[y,x]
    return lab_pixel[0]

def on_mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        brightness = get_brightness(x, y)
        display_image = img.copy()
        cv2.putText(display_image, f'Brightness: {brightness}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Brightness Measurement', display_image)

# Create a window to display the image
cv2.imshow('Brightness Measurement', img)

# Set the mouse move event handler
cv2.setMouseCallback('Brightness Measurement', on_mouse_move)

# polynomial equation for calculation of angle
import numpy as np
from numpy.polynomial import Polynomial

# Example data (replace with your data)
speedometer_readings = np.array([2, 5, 12])
needle_angles = np.array([ 26.56505012512207, 81.20259094238281, 208.663955958252])

# Fit a cubic polynomial to the data
coefficients = np.polyfit( needle_angles,speedometer_readings, 3)

# Print the coefficients of the cubic polynomial
print("Cubic Polynomial Coefficients:", coefficients)
        
# Function for cubic equations: 

def calculate_speed(angle):
      a =-2.20445942e-07    #coefficient[0]
      b =  6.98181121e-05   #coefficient[1]
      c =  4.94678637e-02   #coefficient[2]
      d =   6.40745648e-01  #coefficient[3]
      # 
      # Calculate the speed using the cubic equation
      speed = a * angle**3 + b * angle**2 + c * angle + d  #a,b,c,d-coefficients
      return speed

# testing the speed on the  basis of angle
angle =90 # Replace with your angle value
indicated_speed = calculate_speed(angle)   # Calculate the indicated speed
print(f"Indicated Speed: {indicated_speed} /1000rpm")

# plotting 
reading= np.array([2, 5,12])
Angle=np.array([ 26.56505012512207,81.20259094238281, 208.663955958252])

plt.scatter(reading, Angle , color = 'red', lw = 3)
plt.plot(reading, Angle , color = 'blue', lw = 3)
plt.xlabel('Speedometer-readings------->')
plt.ylabel('Angle--------->')
# img = cv2.putText(img,str(angle), (250,300), 5, 2, (255,255,0), 2, cv2.LINE_AA)
plt.show()
cv2.imshow('Brightness Measurement', img)
# cv2.imwrite("angle_12"+".png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
