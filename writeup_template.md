##Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary_combo_example.png "Binary Example"
[image4]: ./output_images/warped_straight_lines.png "Warp Example"
[image6]: ./output_images/example_output.png "Output"
[video1]: ./output_images/project_video_mapped.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in camera_calibration.py located in the current directory. Method undistort_img calculates the undistored image.

The file, starts by reading all the input camera calibration images, finds object and image points for all the images. This is done in method find_obj_imgpoints. find_obj_imgpoints returns object and image points together for all the calibration images. The object and image points are then used as inputs to cv2.calibrateCamera, whose outputs camera matrix and distortion coefficients are used in cv2.undistort, to get the undistorted image.

The code maps the image points, to the 3D coordinates of the real, undistorted chessboard corners, calling object points. The object points are 3D coordinates, x, y, z from the top-left corner: (0, 0, 0) to (8, 5, 0). 

To create the image points, use OpenCV function called findChessBoardCorners that returns the corners found in a grayscale image.

OpenCV has calibrateCamera, takes in objPoints, imgPoints and shape of the image. Returns, dist: distortion coefficients, camera matrix to transform 3D obj points to 2D image points, position of the camera in the world with values for rvecs, tvecs - rotation and translation vectors. 
I save the object points and image points for later use in test images.

Here is an example of a undistorted image:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Below is a distorted corrected image from the test images folder. See code in `pipeline.py`, undistort_img method.
 
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I read in the saved object and image points, from earlier calibraion step and do the same as above for getting the undisorted image. I do the same as above by calling calibrateCamera, that returns camera matrix and distortion coefficients.  These values are then used in cv2.undistort call.



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

See code `pipeline.py` lines: 402 to 424.
I used a combination of gradient thresholds like sobelx, sobely, magnitude of the gradient, direction of the gradient.
Then I followed up with a color transform using HLS color space. I used S and L components, combined them. The combination of SL and earlier combination of threshold image gives the binary image. 

Here is an example of a binary image:
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In `pipeline.py`, lines: 427, 141 to 149, does the perspective tranform. The method is called `warp()`.
The `warp()` function takes as inputs an image (`img`). Source (`src`) and Destination (`dst`) points are initialized based on the image sizes of original image and an estimation of the destination image.  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32([(200, 720), (580, 480), (720, 480), (1050, 700)])
    dst = np.float32([(280, 720), (400, 290), (920, 290), (960, 720)])
```

`warp()` uses cv2.getPerspectiveTransform, that gives a tranformation matrix for `src` points and `dst` points. This matrix is then used in `cv2.warpPerspective` to generate a perspective transformed image.

Here is a Binary and a Warped image.
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In `pipeline.py`, at lines: 430, methods: 'lanelines', 'lanelines_blindsearch' and 'lanelines_previoussearch' are used to calculate the lane lines. Even though the code uses previous search to optimize the searching of lane lines, I ended up using blindsearch always for every video frame instead of building upon a previous frame. 

lane lines is done this way (I used the code from the lesson):

* get a histogram
* find peak of left and right oh histogram
* choose a sliding window of size=9
* after identifying all the nonzero pixels, determine window boundaries for left and right side.
* After extracting left and right lane pixels, fit a second order polynomial using 'np.polyfit' method for left and right pixel points.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated `radius_of_curvature` in lines 334 through 347 in my code in `pipeline.py`. I converted left and right fit lines from pixels to meters and then calculated radii using the code from the lesson.

I calculated offset from center in method `find_offset_from_center` in lines 385 to 390.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented projection on lane drawing onto the real lane in lines 351 through 383 in my code in `pipeline.py` in the function `project()`.  Here is an example of my result on a test image:


Comments from Review:
"Lanes are drawn on the distorted image instead of undistorted one. It is obvious by looking at the car front hood. Please, make sure to undistort the image in the beginning of the pipeline, use that image for further processing and at the end lanes should also be drawn on the undistorted image not the original image."

Comments from me:
Lane lines are now shown on the undistorted image than on the original image. Although the Project Rubric asked to show on original image. So I am a bit confused.
"The fit from the rectified image has been warped back onto the original image and plotted to identify the lane boundaries."


![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Comments from Review:
"Lanes are drawn on the distorted frame instead of undistorted one."

Comments from me:
Lane lines are now shown on the undistorted frames than on the original frames. 
From the project Rubric here is the criteria:
"Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)"
I believe my video have wobbly lines, but I think not catastrophic. I agree I need to do a better job. I wanted to know if the lane lines are catastrophoic that the Project couldn't be accepted?


Here's a [link to my video result](./output_images/project_video_mapped.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For binary image, I started only with s channel in hls color space. With that the lane wasn't being identified properly. I caputures the bad frames. I experimented on those frames, and found that combining s and l channels gives better results for frames in the shadows. I saw couple of frames doing badly which are not in shadows. The histgram looked good, but for some reason the lanes are identified badly. Found that, mean of right_fit is less than mean of left_fit. In such cases, I used the values from the previous frame. I also added couple of other conditions. I used a blind search of lanes everytime instead of optimizing it for lack of time. I should do a better job with optimizations, tracking previous states using a class like suggested in the lesson. The challenge vidoe performed ok but not perfect. I need to work on having more options to make the challenge video work.