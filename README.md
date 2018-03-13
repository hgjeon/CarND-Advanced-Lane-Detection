**Advanced Lane Finding Project**

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

[image1]: ./output_images/calib_undistort.png "calibration"
[image2]: ./output_images/distortion_correct.png "Undistorted"
[image3]: ./output_images/gradient_threshold.png "Threshold"
[image4]: ./output_images/perspective_transform.png "Perspective transform"
[image5]: ./output_images/lane_detection.png "Line detection"
[image6]: ./output_images/measure_curvature.png "Curvature"

[video1]: ./output_images/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "advanced_lane.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` (3D) will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` (2D) will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-correction

Aforementioned distortion-correction technique has been applied to actual image using the calibrated data from previous procedure. Following two functions have been used.
```python
    cv2.calibrateCamera()
    cv2.undistort()
```
Both original and undistorted images are shown below
![alt text][image2]

#### 2. Color transforms, and Gradients Thresholded Binary Image

I used a combination of color and gradient thresholds to generate a binary image.  
##### Magnitude of Sobel_X and Soble_Y
```python
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = np.sqrt(sobelx * sobelx + sobely * sobely)
```
##### Color (S) Threshold using HLS color space
```python
    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
```

Here's an example of my output for this step.  

![alt text][image3]

#### 3. Perspective Transform

The code for my perspective transform includes a function called `perspectvie_transform(img)` in the file `advanced_lane.ipynb`.  This function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the source from the image, and destination points in the following manner:

```python
src = np.float32([[702, 460], [1127, 720], [203, 720], [578, 460]])

x_max = img.shape[1] * 3 / 4
y_max = img.shape[0] - 0
x_min = img.shape[1] * 1 / 4
y_min = 0
dst = np.float32([[x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Lane Detection

Then, both left and right lines are picked from top two picks from the histogram, and 9 segments of sliding windows have been used to identify those two group of lines. After that 2nd order polynomial curve fit has been used to detect either straight line or curved line.

```python
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```
![alt text][image5]

#### 5. Radius of Curvature of the lane, and the Position of Vehicle

##### Radius of Curvature

The radius of curvature is calculated using the 1st and 2nd derivative of the polynomial.

##### Position of vehicle

The position offset from the center of the road is calculated based on two X-positions at the bottom of image, then this pixel unit has been converted to meter unit.

#### 6. Example image of the result plotted back down onto the road such that the lane area is identified

Once lane detection is done, then these result is converted back to unwarped image using the inverse perspective transform.

```python
newwarp = cv2.warpPerspective(color_warp, perspective_Minv, (image.shape[1], image.shape[0]))
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plt.imshow(result)
```
All this pipeline result for an image is shown below:

![alt text][image6]

---

### Pipeline (video)

#### Sanity checker

The pipeline has been applied to video. At this moment, I added a failure detection using the sanity checker.
1. Check if (MIN_WIDTH < LaneWidth < MAX_WIDTH)
2. Check if sum of difference between current line and previous averaged line positions in Y-axis

#### Error Handling

Once the failure is detected, then I used saved averaged coefficients of last N samples which had good sanity check. Also, N consecutive images are failed by the sanity check, then reset the stored data, and error handling state machine. (N = 8)

#### Video Output  
Here's a [link to my video result][video1]

---

### Discussion

#### 1. Discussion of Problems / Issues

Overall, lane detection is good for entire video. However, it sometimes show a little bit of wobbly line detection from time to time. I would like to implement following to improve this issue.

- Use averaged line detection even though sanity check is passed in current line calculation.
