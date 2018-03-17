
# Parameters of LIDAR scan angles
LIDAR_MIN = -1.69296944141
LIDAR_MAX = 1.6929693222
LIDAR_STEP = 0.00872664619237

# Field of view limit for YOLO
YOLO_FOV_MIN = -0.5
YOLO_FOV_MAX = 0.5

# Farthest range of interest
MAX_R = 10

#################################################################
### SEGMENTATION 
#################################################################

# Size of relevant window for determining pedestrian detection in 
# setting up training data
WINDOW = 6

# Pixels to add on either side of window for trianing
PADDING = 7 

# Total segmnet length
SEGL = WINDOW + 2*PADDING 

# Stride for moving window in training
SEG_STRIDE = 1 


#################################################################
### LOCALIZATION 
#################################################################
# Snap YOLO to closest lidar point for training
TRAIN_SNAP_TO_CLOSEST = False

# Constants for normalizing R and theta
R_BIAS = 0
R_SCALE = 10
TH_BIAS = LIDAR_STEP * PADDING
TH_SCALE = LIDAR_STEP * WINDOW


# Options for localization prediction
USE_LOCNET = True
LOCNET_TYPE = 'polar' # 'polar' or 'cartesian'

# Snap to closest lidar point after prediction
PRED_SNAP_TO_CLOSEST = True


#################################################################
### Evaluation
#################################################################
# Threshold for acceptable ped detection distance
DIST_THRESH = 2

# Shuffle before train/test split
TRAIN_TEST_SHUFFLE = True

# Percent of data to reserve for testing
TEST_SIZE = 0.2 # 0.2

# Percent of training data to reserve for cross-validation
CROSS_VAL_SIZE = 0.2

#################################################################
### Voting/NMS 
#################################################################

# Thresholds used in allowing a vote to be cast
# looepd in precision/recall curve
# THRESHOLDS = np.linspace(0, .98, 10)
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, .97, .975, 0.98, .985, 0.99, .995, 0.999]
# THRESHOLDS = [0.99]

# For pedestrian detection in max suppresion
# Less than 0.5 seems to work
# Larger seems to give slightly better results
# Can also use a very small number and let Gaussian
# filter truncation do the work 
MIN_SCORE = 0.1

# Resolution for voting grid
# Smaller than 0.1 starts to be slow, but smaller
# does have slightly better performance
SCORE_RES = 0.03 

# Parameter for gaussian filter of voting grid
FILTER_SIGMA = 0.1/SCORE_RES # 0.1/SCORE_RES

# Sigma at which to truncate gaussian filter
# 2 seems to work with MIN_SCORE of 0.1
FILTER_TRUNCATE = 2.0

# Minimum distance between predictions after non-max supression
NMS_DIST_THRESH = 0.8*DIST_THRESH

#################################################################
### PLOTTING
#################################################################
# Plot frame/threshold at evaluation step
DO_EVALUATION_PLOT = False

# Plot frame/threshold at thresholding step
DO_THRESHOLD_PLOT = False

# fixed axis limits for animation plots
XLIMS = (-10, 10) 
YLIMS = (-2, 12)

# Animation frames per second
# Real time is about 15
FPS = 30

# Animation pixel density
# 100 for speed
# 300 for quality
DPI = 300

# If True use hand picked clips for animation
# if you want to animate all frames set to False 
USE_SUBSET_FOR_ANIMATION = False

# How many animation sections
# Use None to animate all frames
N_ANIMATION_SECTIONS = 1

# How many consecutive frames to animate for each animation section
ANIMATION_SECTION_WIDTH = 30
    