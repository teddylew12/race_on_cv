# How to Get SetUp
1) Print out one of the tags from the Tags directory <br>
2) Measure them in meters and save those numbers somewhere (mine were .14mx.14m when printed on a normal sheet of paper) <br>
3) Try running the hello world script, make sure you get tag detections from your phone camera <br>
4) Get your camera calibration values from http://www.vision.caltech.edu/bouguetj/calib_doc/ or using OPENCV. <br>
5) Plug those values, and the tag size into the single_image_iphone.py script and see if you get an accurate pose. <br>

# Work to be Done
1) Validate correctness of camera calibration and pose estimation. (Double check values with Fernando)
2) Attempt with more than 1 tag in the photo.
3) Find a way to visualize results?
4) Localization across frame, pose estimation and visual odometry!
