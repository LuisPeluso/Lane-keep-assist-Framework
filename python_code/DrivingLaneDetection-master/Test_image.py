import cv2
import numpy as np
import time; 
ms1 = time.time()
# Please see that the "Marshal to Numpy" option is set on the
# corresponding terminals of the Python node that calls this function from LabVIEW.

def Open_image(img_arr):

	#img_ref = 
	img_arr_out = img_arr#img_arr_out= cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)#cv2.COLOR_GRAY2BGR )
	return img_arr_out;

def image_size (img_arr):
	return (np.asarray(img_arr.shape));
	#return (img_arr.size);
	#print((img_arr.shape))


img_test = cv2.imread("D:\\PELUSO\ITSligo\\lectures_MENG\\5-Thesis\project\\FrameWork\\Test_code\\solidWhiteCurve.jpg")
if img_test is None: raise ValueError("empty image")

print( Open_image(img_test).shape)
ms2 = time.time()
print(ms2-ms1)
#print(image_size(img_test))
a=1