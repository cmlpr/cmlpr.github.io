---
layout: post
title: "Vehicle Detection and Tracking"
date: 2017-03-21
mathjx: true
hlighters: htmlcode
image: vehicle_detection.jpg
imagealt: "Vehicle Detection"
description: "Detect and track cars on video streams using techniques like HOG (Histogram of Oriented Gradients) feature extraction and SVM classifier."
category: Image Processing
tags: image_processing self_driving_cars cars histogram_of_oriented_gradients HOG support_vector_machines svm sliding_window_search vehicle_detection_and_tracking python opencv matplotlib
published: false
comments: true
---

One of the important features of self driving cars is that they are aware of the cars around them. They actually have to detect the cars and track them so that they can react to the changing conditions on the road. One of the methods that can be used to achieve this is to train a classifier by extracting features from car and non-car images. This classifer can then be used to detect whether a section of an image is a car or not. Now let's see how we can do this..

<!--more-->

[//]: # (Image References)

[image1]: /images/vehicle_detection_tracking/spatial_binning1.png "Spatial binning"
[image2]: /images/vehicle_detection_tracking/car_color_hist.png "Car image color histogram"
[image3]: /images/vehicle_detection_tracking/notcar_color_hist.png "Non-car image color histogram"
[image4]: /images/vehicle_detection_tracking/hog_features.png "Hog features"
[image5]: /images/vehicle_detection_tracking/window_search.png "Sliding windows and search"
[image6]: /images/vehicle_detection_tracking/threshold.png "Heatmap and threshold value"
[image7]: /images/vehicle_detection_tracking/threshold2.png "Another heatmap and threshold value"
[image8]: /images/vehicle_detection_tracking/calibration_points_14.png "Sample calibration "
[image9]: /images/vehicle_detection_tracking/calibration_points_14.png "Sample calibration "
[image10]: /images/vehicle_detection_tracking/calibration_points_14.png "Sample calibration "
[image11]: /images/vehicle_detection_tracking/calibration_points_14.png "Sample calibration "
[image12]: /images/vehicle_detection_tracking/calibration_points_14.png "Sample calibration "
[image13]: /images/vehicle_detection_tracking/calibration_points_14.png "Sample calibration "
[image14]: /images/vehicle_detection_tracking/calibration_points_14.png "Sample calibration "
[image15]: /images/vehicle_detection_tracking/calibration_points_14.png "Sample calibration "


---

Here is a high level summary of the process:

* Perform feature extraction on labeled training car and non-car images using color transform, spatial binning and histogram of oriented gradients (HOG)
* Normalize the extracted features and train a linear support vector machines (SVM) classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The code and related files are in this **[REPO](https://github.com/cmlpr/VehicleDetection)**. To make it easier to follow here is the list of files:

| File/Folder             | Definition                                                                         |
| :---------------------- | :----------------------------------------------------------------------------------|
| `VDprocess.py`          | Jupyter notebook with the main process                                             |
| `flib.py`               | Python file with all the helper functions                                          |
| `test_video.mp4`        | Small video to test the pipeline                                                   |
| `project_video.mp4`     | Full video for the pipeline                                                        |
| vehicles                | Folder with 64x64 images of vehicles                                               |
| non-vehicles            | Folder with 64x64 images of non-vehicles                                           |
| test_images             | Folder that includes a number of test images for testing the pipelines             |
| output_images           | Folder that would include any output test images                                   |
|                         |                                                                                    |

---

#### Summary of Helper Functions 

Almost all the function that are used to get features and process data are in the `flib.py` file. 

| File/Folder             | Definition                                                                                        |
| :---------------------- | :-------------------------------------------------------------------------------------------------|
| `draw_boxes`            | Takes an image and a list of rectangle coordinates to draw them on the image                      |
| `color_hist`            | Feature exraction function that compute and get histogram values for each color channel           |
| `bin_spatial`           | Feature extraction function that applies spatial binning to image to reduce size                  |
| `get_hog_features`      | Exracts hog features of an image with for given parameters like orientation, cell and block info  |
| `cs_convert_from_rgb`   | Converts RGB color space to a different color space (HSV, HLS, LUV, YUV, YCrCb)                   |
| `extract_features`      | Calls desired extraction function (color_hist, bin_spatial, get_hog_features) for a list of images and returns features array - will be used for training images|
| `single_img_features`   | Same as extract_features function but works for only one image - will be used on video images |
| `slide_window`          | Extracts windows from an image - size, overlap, start/stop positions |
| `search_windows`        | Loops through each extracted window, get features and call classifier to predict if the window has a car in it |
| `find_cars`             | A more efficient way of finding cars in images - instead of calculating hog features for extracted window images, computes hog features once for the entire image and slides the window on a scaled image for prediction |
| `apply_threshold`       | Removes values below a threshold from a heatmap |
| `draw_labeled_bboxes`   | Draws labeled bounding boxes on an image |
| `visualize`             | Plots various images |
| `color_histogram`       | Plots color channel histograms | 
| `process_video`         | Function to process video |
| `process_image_*`       | Main pipeline function that processes image - there are multiple of them from one to five |
|                         |                                                                                    |

---

#### Data

Since we will be identifying cars using a classification algorithm, we need to train the classifier using car and non-car images. These images are located in two different folders `vehicles` and `non-vehicles` with subfolders. These example images come from a combination of the `GTI vehicle image database`, the `KITTI vision benchmark suite`, and examples extracted from the project video itself.

We can use glob package to get various `png` images under different subfolders using a pattern.

```python
car_image_paths = glob.glob('vehicles/*/*.png')
notcar_image_paths = glob.glob('non-vehicles/*/*.png')

# Now let's see the total number of images we have in each class 
print('Number of Vehicle Images found: ', len(car_image_paths))
print('Number of Non-Vehicle Images found: ', len(notcar_image_paths))

Number of Vehicle Images found:  8792
Number of Non-Vehicle Images found:  8968
```

The examples in each class is close which is nice from classifier training perspective. 

---

#### Feature Extraction

**Spatial Binning**

The simplest information we can get from an image is the pixel values. However we usually reduce the size of the image to minimize the data while keeping enough of it to distinguish characteristic features. `bin_spatial` function in `flib.py` file uses `resize` function in `openCV` to resize the image with a given bin size. Here are examples from car and non-car images after applying spatial binning with 16x16 bin size:

![Spatial binning][image1]{: .post_img }

**Color Histograms**

Frequency or count information of each pixel value in each channel can also be used to detect objects in an image. We hope to see different histogram data for car and not-car images for better classification. `flib.py` file includes the `color_hist` function that takes an image and returns channel histograms, bin centers and a flat concatenated feature vector for classification. 

Histogram of the car image shown above: 

![Histogram of car image][image2]{: .post_img }

Histogram of the non-car image shown above: 

![Histogram of non-car image][image3]{: .post_img }

As can be seen from these histograms color values can occur at different frequencies for various shapes. 

**Histogram of Oriented Gradients (HOG) for Gradient Features**

Gradients and edges can give us more robost representation. The presence of gradients of in specific directions around center can capture shape information. HOG method has been developed to do just that. You can read more about HOG feature detection algorithm in **[Wikipedia](https://github.com/cmlpr/VehicleDetection)**. HOG is robust to variations in shape while keeping the signature distinct enough. `get_hog_fetures` function in `flib.py` file calls `hog` function from `scikit-image` package with a list of parameters. Some of these parameters are number of `orientations`, `pixels per cell` and `cells per block`. The function also return the processed image if requested.  Here is a visualization of hog features with parameters: 

orient = 6

pix_per_cell = 8

cell_per_block = 2

hog_channel = 0

![HoG features][image4]{: .post_img }

When car and non-car images are compared with their HOG feature images, it becomes obvious that HOG method captures some of the distinct features of the car. Consider an image showing the back of the car, vectors shows that HOG can capture the licence plate, car boundary, rear window, lights, etc. The two parameters pix_per_cell and cell_per_block are selected based on the size of the features we have in the image. 

**Feature extraction helper function**

There are two other functions in `flib.py` that helps me to extract a full set of features from images. The first one is the `extract_features` function. This function is designed to extract features from training car and non-car images. It takes a list of training image locations and calls each feature extraction function described above if desired. This function returns a list of feature vectors. The second function is `single_img_features`. When predicting car/not-car for individual windows we will use this function. The content of this function is the same as previos one with the nuance that it only works for one image. 

---

#### Model Training 

All the images in `vehicle` and `non-vehicle` folders are used in the feature extraction for model training. Instead of `RGB` color space `YCrCb` color space which makes it easy to get rid of some redundant color information is used in the color histograms. `(32, 32)` bin size for spatial binning was found to generate a better test accuracy during model training. HOG parameters were mostly trial and error with some guidance on values. As number of orientations is increased we expect to capture more details; however if the make it too high there will be more noise in the data for smaller features which indicates that `signal-to-noise` ratio will decrease. 9 orientations found to be good. `pixels_per_cell` specifies the cell size over which each gradient histogram is computed. 8 pixels in 64x64 images generates good results. `cell_per_block` is the area over which the histogram counts in a given cell will be normalized. Both `pixels_per_cell` and `cell_per_block` are specified based on the size of the features we have in the image. 

| Parameter      | Value         | 
|:------------- :|:-------------:| 
| color_space    | YCrCb         | 
| orient         | 9             |
| pix_per_cell   | 8             |
| cell_per_block | 2             |
| hog_channel    | ALL           |
| spatial_size   | (32, 32)      |
| hist_bins      | 32            |
| spatial_feat   | True          |
| hist_feat      | True          |
| hop_feat       | True          |
|                |               |

Since resulting feature vectors have 3 sections from 3 feature extraction methods which yields different magnitudes, it is best to normalize the data. `StandardScaler` from `sklearn` machine learning package is very easy to use in this case. 

The next step is to shuffle and split the data into training and test sets. `shuffle` and `train_test_split` functions are used for these processes respecitvely. 

For classification `Support Vector Machines (SVM)` with `linear` kernel is good enough for this process. Default parameters are accepted while training the data. It would be interesting to test various parameters or classification models but for now I won't be spending more time on the classification part. 

The section of the code for training can be found in `VDprocess.ipynb` notebook and in the section below: 

```python
# Define feature parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8 # size of the features we are looking in the images 
cell_per_block = 2 # helps with the normalization - lighting, shadows
hog_channel = 'ALL'  # can be 0, 1, 2, or 'ALL' 
spatial_size = (32, 32)  # Spatial binning dimensions 
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Get spatial features on/off
hist_feat = True  # Get color histogram features on/off
hog_feat = True  # Get HOG features on/off

t = time.time()

test_cars = car_image_paths
test_notcars = notcar_image_paths

car_features = flib.extract_features(test_cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block, hog_channel=hog_channel,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features = flib.extract_features(test_notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block, hog_channel=hog_channel,
                                   spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

print(round(time.time() - t, 2), 'Seconds to compute the features\n')

X = np.vstack((car_features, notcar_features)).astype(np.float64) # Standard Scaler expects float 64
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Normalize Data
# We have spatial, color histogram and HOG features in the same feature set
# It is best to bring them to equal scale to avoid one feature to dominate due to scale differences
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Use the scaler to transform X 
scaled_X = X_scaler.transform(X)

# Shuffle Data
scaled_X, y = shuffle(scaled_X, y)

# Split the data into train and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', 
      cell_per_block, 'cells per block', hist_bins, 'histogram bins, and', spatial_size, 'spatial sampling\n')
print('Feature vector length:', len(X_train[0]), '\n')

# Use linear SVC
svc = LinearSVC()

# Check the training time for the SVC
t = time.time()

svc.fit(X_train, y_train)

print(round(time.time() - t, 2), 'Seconds to train SVC..\n')

# Check the score of the SVC
svc_score = svc.score(X_test, y_test)

print('Test accuracy of SVC = ', round(svc_score, 4), '\n')

82.08 Seconds to compute the features

Using: 9 orientations 8 pixels per cell and 2 cells per block 32 histogram bins, and (32, 32) spatial sampling

Feature vector length: 8460 

22.13 Seconds to train SVC..

Test accuracy of SVC =  0.9935 
```

Test accuracy looks ok for a quick classification work. 


---

#### Sliding Window Search 

The goal is to come up with a pipeline to process video images and detect/track cars. So each frame in a video needs to be analyzed and searched for cars. One way of doing this is to use sliding window approach on images to split the image into many windows and perform predictions for each window. These functions are in `flib.py` with names `slide_window` and `search_windows`. 

Searching bottom portion of the image (between y = 400px and y = 656px - rest of the image is not interesting) we can test example images that are in `test_images` folder. For this test I picked 3 different windows sizes `48`, `64` and `128` with 50% overlap. The parameters yields 798 windows per image and it took ~2.8s to process an image which seems to be high. One thing we could do is to reduce the windows size and change the overlap to reduce total windows processed for an image. 

```python
example_images = glob.glob('test_images/test*.jpg')
images = []
titles = []
x_start_stop = [None, None]
y_start_stop = [400, 656]  # Min and max in y to search in slide_window()
overlap = 0.5
window_size_small = 48
window_size_medium = 64
window_size_large = 128
viz=False

for img_path in example_images:
    
    print(img_path)
    t1 = time.time()
    img = mpimg.imread(img_path)
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255  # we developed the pipeline for png images the ranges for it 0-1
    # print(np.min(img), np.max(img))
    
    windows_small = flib.slide_window(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                           xy_window=(window_size_small, window_size_small), xy_overlap=(overlap, overlap))
    
    windows_medium = flib.slide_window(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                           xy_window=(window_size_medium, window_size_medium), xy_overlap=(overlap, overlap))
    
    windows_large = flib.slide_window(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                           xy_window=(window_size_large, window_size_large), xy_overlap=(overlap, overlap))
    
    windows = windows_small + windows_medium + windows_large
    
    hot_windows = flib.search_windows(img, windows, svc, X_scaler,  
                                 color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, 
                                 orient=orient, pix_per_cell=pix_per_cell, 
                                 cell_per_block=cell_per_block, hog_channel=hog_channel, 
                                 spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    
    window_img = flib.draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
    images.append(window_img)
    titles.append('')
    print(round(time.time() - t1, 2), 'seconds to process one image searching', len(windows), 'windows')

fig = plt.figure(figsize=(12, 12), dpi = 300)
flib.visualize(fig, 3, 2, images, titles)

test_images/test1.jpg
2.85 seconds to process one image searching 798 windows
test_images/test2.jpg
2.74 seconds to process one image searching 798 windows
test_images/test3.jpg
2.72 seconds to process one image searching 798 windows
test_images/test4.jpg
2.87 seconds to process one image searching 798 windows
test_images/test5.jpg
2.8 seconds to process one image searching 798 windows
test_images/test6.jpg
2.76 seconds to process one image searching 798 windows

```

![Search Windows][image5]{: .post_img }

This process seems to work but except that it takes significant time and there are a number of false positive and lots of overlap. To overcome these problems we can introduce a few new functions/methods.

* `find_cars`: This function is an efficient implementation of sliding window search. In previous sliding window search the most time consuming part was calculating the hog features for each window. This process can be significantly made faster by computing the HOG features for the entire image and sliding along the feature array to subset the section of the HOG features array for the desired window position. Additionally this function uses a scaling parameter `scale` instead of `window_size`. The functions scales the original image based on the parameter value instead of asking for a window size. Effectively both techniques are doing the same thing. Another addition in this function is the `heatmaps`. When a window is detected, we can add heat (ones) to an 2-D array zeros. After all windows are analyzed, we will have a nice heatmap from which we can determine the bounding boxes of the heat zones. This method will reduce the overlap and will help us combine windows detected using various scales. 

* `apply_threshold`: Function to remove small values (noise) from heatmaps. When multiple consecutive frames are analyzed and averaged we can apply this function to remove the noise and reduce the number of false positives. 

* `label` from `scipy.ndimage.measurements`: Helps us detect heated areas from the heatmaps

* `draw_labeled_bboxes`: Get labels found by the `label` function and an image, and returns image with bounding boxes. 

To generate a pipeline with these functions and select parameters that would work for the video, we can extract more images fromt he video and process them with `process_image_four` function which is in `VDprocess.ipynb` notebook. 

```python
vidcap = cv2.VideoCapture('project_video.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  count += 1
    
print(count)

1260
```

We have `1260` frames in the video. It would be ok to extract every 50th frame. 

```python
vidcap = cv2.VideoCapture('project_video.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
    success, image = vidcap.read()
    if not success:
        print('Problem reading frame: ', count)
    if count % 50 == 1:
        cv2.imwrite("test_images/prj_video_frame%d.jpg" % count, image)     # save frame as JPEG file
    count += 1
```

Now we can define our x and y start/stop positions, scales (we will average the heatmaps from each scale) and threshold for false-positives:

```python
some_prj_imgs = glob.glob('test_images/prj_video_frame*.jpg')

x_start_stop = [450, None]
y_start_stop = [400, 656]  
scales = [1.0, 1.5, 2.0]
threshold = 0.5

def process_image_four(img):

    heat_maps = []
    for scale in scales:
        out_img, out_ = flib.find_cars(img=img, scale=scale, x_start_stop=x_start_stop,
                                         y_start_stop=y_start_stop, clf=svc, scaler=X_scaler,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, spatial_size=spatial_size,
                                         hist_bins=hist_bins, color_space=color_space)
        heat_maps.append(out_)
    avg_ = np.divide(np.sum(heat_maps, axis=0), len(scales))
    final_ = flib.apply_threshold(avg_, threshold)
    labels = label(final_)
    draw_img = flib.draw_labeled_bboxes(np.copy(img), labels)
    return final_, draw_img

for img_path in some_prj_imgs:
    
    img = mpimg.imread(img_path)
    himg, img_box = process_image_four(img)
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,4))
    f.subplots_adjust(hspace=2.0)
    
    ax1.imshow(img)
    ax2.imshow(img_box)
    ax3.imshow(himg, cmap='hot')
    ax4.hist(himg.ravel()[np.flatnonzero(himg)], 50, normed=1, facecolor='green', alpha=0.75)

    plt.show()
    plt.close()
```

Some example outputs are shown below. More examples are in the ipython notebook. It looks like we could get a pretty decent result. 

![Thresholded heatmap][image6]{: .post_img }

![Another Thresholded heatmap][image7]{: .post_img }

The next step is to modify `process_image_four` pipeline function to store historical heatmaps for averaging. This way we could get more consitent bounding boxes for cars. We define two global variables (`global_heatmaps` list and `heatmap_sum`) and update these variables in the pipeline function `process_image_five`. We store last 8 heatmap images, sum them up and take the average to get smoother data. When we get a new heatmap we have to subtract the first heatmap so that we don't get a streak of heatmaps. When subtracting we might end up with negative values, therefore the `heatmap_sum` will be cliped to values between 0 and a large number. 

```python

global_heatmaps = []
heatmap_sum = np.zeros((720,1280)).astype(np.float64)


def process_image_five(img):
    
    global global_heatmaps, heatmap_sum
    
    heat_maps = []
    
    for scale in scales:
        out_img, out_heatmap = flib.find_cars(img=img, scale=scale, x_start_stop=x_start_stop,
                                              y_start_stop=y_start_stop, clf=svc, scaler=X_scaler,
                                              orient=orient, pix_per_cell=pix_per_cell,
                                              cell_per_block=cell_per_block, spatial_size=spatial_size,
                                              hist_bins=hist_bins, color_space=color_space)
        heat_maps.append(out_heatmap)
    
    local_heatmap = np.divide(np.sum(heat_maps, axis=0), len(scales))
    
    global_heatmaps.append(local_heatmap)
    heatmap_sum += local_heatmap
    
    if len(global_heatmaps) > 8:
        oldest_heatmap = global_heatmaps.pop(0)
        heatmap_sum -= oldest_heatmap
        heatmap_sum = np.clip(heatmap_sum, 0.0, 1000000.0)
        
    heatmap_avg = heatmap_sum / len(global_heatmaps)
    
    final_heatmap = flib.apply_threshold(heatmap_avg, threshold)
    labels = label(final_heatmap)
    draw_img = flib.draw_labeled_bboxes(np.copy(img), labels)
    return draw_img
```

New prefered parameters are:

x_start_stop = [450, None] - left side of the image is removed to eliminate detection of the opposite lane
 
y_start_stop = [400, 656]  - region of interest for rows

scales = [1.0, 1.5, 2.0] - three scales are used to find cars 

threshold = 0.3 - small 


Call the pipeline for small video clip

```python
flib.process_video("test_video.mp4", "test_out.mp4", process_image_five)
```

<div class="post_videoWrapper">
    <!-- Copy & Pasted from YouTube -->
    <iframe src="https://www.youtube.com/embed/Upo7kldnKAM" frameborder="0" allowfullscreen></iframe>
</div>

Reinitialize the global values and process the full video

```python
global_heatmaps = []
heatmap_sum = np.zeros((720,1280)).astype(np.float64)

flib.process_video('project_video.mp4', 'project_out.mp4', process_image_five)
```

<div class="post_videoWrapper">
    <!-- Copy & Pasted from YouTube -->
    <iframe src="https://www.youtube.com/embed/uXlXqXk9j_g" frameborder="0" allowfullscreen></iframe>
</div>


---

#### Discussion

Feature extraction, SVM classifier with linear kernel and sliding window techniques were used to detect and track vehicles on a video stream. Although this first take on the vehicle detection methodology yielded OK result, there are many aspects that could be improved. Here are a list of them:

* Identify each car and prevent car overlapping. This can be performed by keeping track of previous bounding boxes of individual cars and subtracting their moving averege from newly processed heatmaps. 
* Model training: Obtain and use more data; with limited data detecting cars in the city could be challenging. One can try different model paramaters for the SVM and apply Cross Validation (CV). How about other classification techniques?
* Fine tune start/stop positions (may not work at different slopes), scale and threshold values for other videos with different background (such as city). One can split the image into sections where smaller objects are searched in distant locations and larger obects are searched otherwise. 
* Combine vehicle detection with lane line detection so get warnings for proximity or forward collusion. 

