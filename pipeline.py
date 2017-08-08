import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import pickle
from moviepy.editor import VideoFileClip


def undistort_img(img, points3d, points2d):
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points3d, points2d, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


def abs_sobel_thres(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative
    orient_x = 1 if (orient == 'x') else 0
    orient_y = not orient_x
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient_x, orient_y, ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output



def mag_thresh(img, sobel_kernel=3, mag_thres=(0, 255)):
    # calculate gradient magnitude
    # Apply threshold
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x orient
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * gradmag / np.max(gradmag  ))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thres[0]) & (scaled_sobel <= mag_thres[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # calculate gradient direction
    # Apply threshold

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x orient
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(grad_direction)
    binary_output[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

def hls_s_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hlsimg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    s_channel = hlsimg[:, :, 2]

    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(s_channel)

    # 3) Return a binary image of threshold result
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    return binary_output

def hls_l_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hlsimg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    l_channel = hlsimg[:, :, 1]

    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(l_channel)

    # 3) Return a binary image of threshold result
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1

    return binary_output


def hls_h_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hlsimg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    h_channel = hlsimg[:, :, 0]

    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(h_channel)

    # 3) Return a binary image of threshold result
    binary_output[(h_channel > thresh[0]) & (h_channel <= thresh[1])] = 1

    return binary_output


from random import randint
def warp(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([(200, 720), (580, 480), (720, 480), (1050, 700)])
    dst = np.float32([(280, 720), (400, 290), (920, 290), (960, 720)])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M

def display(orig, dst):
    src = orig #cv2.imread(orig)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(orig) #cv2.cvtColor(src, cv2.COLOR_RGB2BGR))
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst, cmap='gray')
    ax2.set_title('Binary', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def display2(src, dst):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(src, cmap='gray')
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(dst, cmap='gray')
    ax2.set_title('Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def lanelines_previoussearch(img, left_fit, right_fit):
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 100

    left_lane_inds = []
    right_lane_inds = []

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] +margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    avg_left_fitx = np.mean(left_fitx)
    avg_right_fitx = np.mean(right_fitx)

    return ploty, left_fit, right_fit, left_fitx, right_fitx, avg_left_fitx, avg_right_fitx

saved_ploty = None
saved_left_fit = None
saved_right_fit = None
saved_left_fitx = None
saved_right_fitx = None
saved_avg_left_fitx = None
saved_avg_right_fitx = None

def lanelines_blindsearch(img):
    # Take histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
    # plt.plot(histogram)
    # plt.show()

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lanes
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    avg_left_fitx = np.mean(left_fitx)
    avg_right_fitx = np.mean(right_fitx)

    center_of_image = img.shape[1] / 2

    global saved_ploty, saved_left_fit, saved_right_fit, saved_left_fitx, saved_right_fitx, saved_avg_left_fitx, saved_avg_right_fitx
    if (np.mean(right_fit) > np.mean(left_fit) and np.mean(right_fit) > np.mean(righty)) or (saved_ploty is None):
        saved_ploty = ploty
        saved_left_fit = left_fit
        saved_right_fit = right_fit
        saved_left_fitx = left_fitx
        saved_right_fitx = right_fitx
        saved_avg_left_fitx = avg_left_fitx
        saved_avg_right_fitx = avg_right_fitx

    return saved_ploty, saved_left_fit, saved_right_fit, saved_left_fitx, saved_right_fitx, saved_avg_left_fitx, saved_avg_right_fitx


left_fit_current = None
right_fit_current = None

def lanelines(img):
    global left_fit_current
    global right_fit_current

    if (left_fit_current is None) or (right_fit_current is None):
        ploty, left_fit, right_fit, left_fitx, right_fitx, avg_left_fitx, avg_right_fitx = lanelines_blindsearch(img)
    else:
        ploty, left_fit, right_fit, left_fitx, right_fitx, avg_left_fitx, avg_right_fitx = lanelines_previoussearch(img, left_fit_current, right_fit_current)

    left_fit_current = None #left_fit
    right_fit_current = None #right_fit

    return ploty, left_fitx, right_fitx, avg_left_fitx, avg_right_fitx


def radius_of_curvature(ploty, left_fitx, right_fitx):
    y_eval   = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad



def project(warped, img, ploty, left_fitx, right_fitx, M, avg_curverad, offset, avg_left_fitx, avg_right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = np.linalg.inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature: {} m".format(int(avg_curverad))
    cv2.putText(result, text, (0, 50), font, 1, (255, 255, 255), 3)

    if (offset < 0):
        text = "Vehicle is {:.2f} m left of center".format(-offset)
    else:
        text = "Vehicle is {:.2f} m right of center".format(offset)

    cv2.putText(result, text, (0, 100), font, 1, (255, 255, 255), 3)
    # plt.imshow(result)
    # plt.show()
    return result

def find_offset_from_center(img, left_fitx, right_fitx):
    center_of_image = img.shape[1]/2
    center_of_lane = np.mean((left_fitx + right_fitx)/2)
    xm_per_pix = 3.7/700  # meters per pixel in x dimension
    offset = (center_of_image - center_of_lane) * xm_per_pix
    return offset


with open('wide_dist_pickle.p', 'rb') as f:
    objpoints = pickle.load(f)
    imgpoints = pickle.load(f)

# Choose a sobel kernel size
ksize = 7
def pipeline(orig):
    img = undistort_img(orig, objpoints, imgpoints)

    gradx = abs_sobel_thres(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thres(img, orient='y', sobel_kernel=ksize, thresh=(40, 90))

    mag_binary = mag_thresh(img, sobel_kernel=9, mag_thres=(30, 100))

    dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.2))

    combined_grad_mag_dir = np.zeros_like(dir_binary)
    combined_grad_mag_dir[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # display(img, combined_grad_mag_dir)

    s_binary = hls_s_select(img, thresh=(120, 240))
    # display(img, s_binary)

    l_binary = hls_l_select(img, thresh=(120, 240))
    # display(img, l_binary)

    combined_sl = np.zeros_like(s_binary)
    combined_sl[(s_binary == 1) & (l_binary == 1)] = 1
    # display(img, combined_sl)

    combined_grad_mag_dir_hls = np.zeros_like(combined_sl)
    combined_grad_mag_dir_hls[(combined_grad_mag_dir == 1) | (combined_sl == 1)] = 1
    # display(img, combined_grad_mag_dir_hls)

    warped, M = warp(combined_grad_mag_dir_hls)
    # display2(combined_grad_mag_dir_hls, warped)

    ploty, left_fitx, right_fitx, avg_left_fitx, avg_right_fitx = lanelines(warped)
    left_curverad, right_curverad = radius_of_curvature(ploty, left_fitx, right_fitx)
    avg_curverad = (left_curverad + right_curverad)/2

    offset = find_offset_from_center(img, left_fitx, right_fitx)

    result = project(warped, img, ploty, left_fitx, right_fitx, M, avg_curverad, offset, avg_left_fitx, avg_right_fitx)
    return result



test_images_path = 'test_images/'
images = glob.glob(test_images_path+"*.jpg")
for image in images:
    orig = mpimg.imread(image)
    left_fit_current = None
    right_fit_current = None
    pipeline(orig)


white_output = 'output_images/project_video_mapped.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline)
white_clip.write_videofile(white_output, audio=False)
