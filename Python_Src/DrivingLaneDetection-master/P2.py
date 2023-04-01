# %% [markdown]
# # **Finding Lane Lines on the Road** 
# 
# When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are, act as our constant reference for where to steer the vehicle. 
# 
# This project proposes an algorithm to automatically detect lane lines using OpenCV and Python, including Hough Transforms and Canny edge detection. Detected highway lane lines on a video stream. Ultimately, we would draw just one line for the left side of the lane, and one for the right.
# 
# **The tools I have used are as follows: color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection, Hough Tranform line detection, and average windows for lane positions.**

# %% [markdown]
# Helper functions are listed as follows:

# %%
#importing some useful packages
import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
#%matplotlib inline

# in order to get stable lane, buffer N frames' slopes and intercepts
pre_l_slopes = []
pre_l_inters = []  
pre_r_slopes = []
pre_r_inters = []


def clear_buffer():
    """
    rest the lines buffer
    :return:
    """
    pre_l_slopes.clear()
    pre_l_inters.clear()
    pre_r_slopes.clear()
    pre_r_inters.clear()
    return 1 ;

def selectWhiteAndYellow(img):
    """
    selec the white and yellow component in the hsv space.
    (1) set the yellow/white lower and upper bound
    (2) apply the mask to the hsv space image
    """
    lower_yellow = np.array([50, 90, 0], np.uint8)#[65, 100, 100]
    upper_yellow = np.array([255, 255, 254], np.uint8)#[105, 255, 255]
    yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([254, 255, 255])  # range for H is 0:179 [179, 20, 255]
    white_mask = cv2.inRange(img, lower_white, upper_white)

    img = cv2.bitwise_and(img, img, mask=yellow_mask)#cv2.bitwise_or(yellow_mask, white_mask))
    return img

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #return mask;
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def outlierCleaner(predictions, x, y, inlier_percent=0.9):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual y values).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (x, y, error).
    """
    threshold = 4  # if the biggest error is greater than 5 pixels, we perform outliers remove
    errs = np.fabs(y - predictions)
    max_err = max(errs)[0]
    if max_err > threshold:  # if the biggest error is greater than 5 pixels, we remove the outliers
        k = math.ceil(errs.size * inlier_percent)
        survived_idx = np.argsort(errs, axis=0)[:k]  # find the number of k min errs, and return their index
        if survived_idx.size > 0:
            x = np.take(x, survived_idx)
            y = np.take(y, survived_idx)

    return (x, y)

def findTwoPoints(slope, inter, side, height):
    """
    In order to get two stable lanes,
    (1) average the slope and itercept values in the buffers, 
    (2) fix the y coordinate of the top points
    (3) then use the averaged slope and inter to locate the two end points of a line

    :param slope: current slope from the regressor for current frame
    :param inter: current intercept from the regressor for current frame
    :param side:  'l': left, 'r': right lane
    :param height: hight of the image
    :return tow points locations, which are the two ends of a lane
    """
    number_buffer_frames = 3
    scale_y = 0.48#0.65
    top_y = int(float(height) * scale_y)  # fix the y coordinates of the top point, so that the line is more stable
    

    if side == 'l':
        if len(pre_l_slopes) == number_buffer_frames:  # reach the max
            pre_l_slopes.pop(0)  # remove the oldest frame
            pre_l_inters.pop(0)

        pre_l_slopes.append(slope)
        pre_l_inters.append(inter)
        slope = sum(pre_l_slopes) / len(pre_l_slopes)
        inter = sum(pre_l_inters) / len(pre_l_inters)

        p1_y = height-1
        p1_x = int((float(p1_y)-inter)/slope)
        p2_y = top_y
        p2_x = int((float(p2_y)-inter)/slope)
    else:  # 'r'
        if len(pre_r_slopes) == number_buffer_frames:  # reach the max
            pre_r_slopes.pop(0)  # remove the oldest frame
            pre_r_inters.pop(0)

        pre_r_slopes.append(slope)
        pre_r_inters.append(inter)
        slope = sum(pre_r_slopes) / len(pre_r_slopes)
        inter = sum(pre_r_inters) / len(pre_r_inters)

        p1_y = top_y
        p1_x = int((float(p1_y)-inter)/slope)
        p2_y = height-1
        p2_x = int((float(p2_y)-inter)/slope)

    return (p1_x, p1_y, p2_x, p2_y)


def regress_a_lane(img, x, y, color=[255, 0, 0], thickness=10):#lane overlay thickness
    """ regress a line from x, y and add it to img
    (1) use a linear regressor to fit the data (x,y)
    (2) remove outlier, and then fit the cleaned data again to get slope and intercept
    (3) find the two ends of the desired line by using slope and intercept
    
    :param img: input image
    :param x: x coordinate
    :param y: y coordinate
    :param color: line color
    :param thickness: thickness of the line  
    """
    reg = LinearRegression()
    reg.fit(x, y)

    # identify and remove outliers
    cleaned_data = []
    try:
        predictions = reg.predict(x)
        cleaned_data = outlierCleaner(predictions, x, y)
    except NameError:
        print("err in regression prediction")

    if len(cleaned_data) > 0:
        x, y = cleaned_data   
        # refit cleaned data!
        try:
            reg.fit(x, y)
        except NameError:
            print("err in reg.fit for cleaned data")
    else:
        print("outlierCleaner() is returning an empty list, no refitting to be done")

    height = img.shape[0]
    slope = reg.coef_
    inter = reg.intercept_

    # find the two end points of the line by using slope and iter, and then visulize the line
    if reg.coef_ < 0:  # left lane
        p1_x, p1_y, p2_x, p2_y = findTwoPoints(slope, inter, 'l', height)
        cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), color, thickness)
    else:  # right lane
        p1_x, p1_y, p2_x, p2_y = findTwoPoints(slope, inter, 'r', height)
        cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), color, thickness)

    return [p1_x, p1_y, p2_x, p2_y]

def draw_lines(img, lines, color=[255, 0, 0], thickness=15):
    """
    (1) remove some horizontal lines with given threshold_angle
    (2) seperate the points belongs to the left and right lane by computing line slope
    (3) handle the left/right lane points to a linear regressor to fit the line, with additional 
        steps to remove the outliers for getting a better fit.
    """
    threshold_angle = 25#25  # if the line angle is between -25 to 25 degrees, lines are discarded
    threshold_slope = math.tan(threshold_angle / 180 * math.pi)
    left_lane_x = []
    left_lane_y = []
    right_lane_x = []
    right_lane_y = []

    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 != x1:
                slope = float(y2 - y1) / float(x2 - x1)
                if abs(slope) < threshold_slope:  # remove the horizontal lines
                    continue
                elif slope < 0:  # left lane, note the origin is on the left-up corner of the image
                    left_lane_x.append([x1])
                    left_lane_y.append([y1])
                    left_lane_x.append([x2])
                    left_lane_y.append([y2])
                else:
                    right_lane_x.append([x1])
                    right_lane_y.append([y1])#img_Consumer_SUT_23
                    right_lane_x.append([x2])
                    right_lane_y.append([y2])

    # get left and right solid lanes with regression
    Left_lane_pts = []
    Right_lane_pts = []
    if len(left_lane_x) > 0:  # if there are no enough points at the current frame
        Left_lane_pts= regress_a_lane(img, left_lane_x, left_lane_y)
    if len(right_lane_x) > 0:
        Right_lane_pts= regress_a_lane(img, right_lane_x, right_lane_y)

    return [[Left_lane_pts] , [Right_lane_pts]] #should mathc 3D array

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    Lane_pts = []
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if (lines is not None):
        Lane_pts = draw_lines(line_img, lines)

    return line_img, lines, Lane_pts #lines   #


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.7, β=1., λ=0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

# %% [markdown]
# ## Image Processing Pipeline for lane detection and visualization
# see details in function drawLanesPipeline

# %
def image_shape(image=[]):
    return (image.shape);

def set_parameters(Kernel_size,):
    Kernel
    return;

#---------set parameters----------#
# gaussian_blur para
kernel_size = 3#5
# canny edege detection para
low_threshold = 25#50
high_threshold = 150#150

# scale_w = 7/16#7 / 16
# scale_h = 11.8/18#13/18#11 / 18
# offset_X = 50
# offset_Y=50


# min_line_length = 15#15  # minimum number of pixels making up a line
# max_line_gap = 40#40  # maximum gap in pixels between connectable line segments

def SetParameters (a,b,c,d, r, t, th, mll, mlg):
    global scale_w
    global scale_h
    global offset_X
    global offset_Y

    # hough_line para
    global rho 
    global theta 
    global threshold
    global min_line_length
    global max_line_gap

    scale_w = a
    scale_h = b
    offset_X = c
    offset_Y = d

    rho = r#1#2  # distance resolution in pixels of the Hough grid
    theta = t#(np.pi / 180)/2  # angular resolution in radians of the Hough grid
    threshold = th#12#12#15  # minimum number of votes (intersections in Hough grid cell)

    min_line_length = mll
    max_line_gap = mlg

    return a,mlg
#---------------------------------#


def drawLanesPipeline(image):
    #image = image.astype(np.uint8)
    """ Process one image, detect two lanes and highlight them with solid color lines
    (1) apply the gaussian blur
    (2) convert bgr to hsv and segment while and yellow, because it is easier in HSV space than RGB
    (3) Canny edge detection
    (4) apply the designed mask to the image to obtian the region of interest
    (5) apply hough transform to get lines
    (6) augmented the lanes on the original image
    
    :param image: input image
    :return: an augmented image with two lane highlighted
    """
    # region_of_interest para
    height,width,_ = image.shape
    left_bottom = [offset_X, height - offset_Y]
    right_bottom = [width - offset_X, height - offset_Y]
    left_up = [scale_w * width, scale_h * height]
    right_up = [(1 - scale_w) * width, scale_h * height]
    
    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(image, kernel_size)
    
    # convert image from bgr to hsv
    #hsv_img = cv2.cvtColor(blur_gray, cv2.COLOR_RGB2HSV)#COLOR_BGR2HSV
   
    # filter out the white and yellow segments (lanes are either white or yellow in this case)
    filtered = selectWhiteAndYellow(blur_gray)#hsv_img)
    #return filtered_hsv
    # Apply Canny edge detection
    edges = canny(filtered, low_threshold, high_threshold)
    #return edges;
    # create a masked edges image
    vertices = np.array([[left_bottom, left_up, right_up, right_bottom]], dtype=np.int32)
    #vertices = np.array([[right_up,right_bottom,left_bottom, left_up]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    #return masked_edges;
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image, Lines, lanes_points = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the edge image
    output = blur_gray#blur_gray#edges#weighted_img(line_image, image)

    str_Lines = str(Lines)
    
    return output, vertices, masked_edges, Lines, lanes_points   #str_Lines#Lines  np.array([[[0]]]), np.array([[0]],dtype=np.uint8), np.array([[[0]]])#

# %% [markdown]
# ## Test on Images
# 
# Now you should build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# %% [markdown]
# run your solution on all test_images and make copies into the test_images directory).

# %%
'''
for fn in os.listdir('D:/PELUSO/ITSligo/lectures_MENG/5-Thesis/project/FrameWork/Test_code/'):
    fileName, ending = os.path.splitext(fn)
    if ending != ".jpg":
        continue

    image = mpimg.imread(os.path.join("test_images", fileName + ending))
    
    clear_buffer()
    # draw two lanes on the image
    image_augmented = drawLanesPipeline(image) 
    a =1  
'''
