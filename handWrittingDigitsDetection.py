# Course Project @ ND computer vision (2017)
# Boyang Li
# ----------------------------------

# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
from collections import Counter
import cv2
import argparse as ap

def segmNtrain():
    print("seg")
#////////////////////////////////////////
# Load the dataset
    dataset = datasets.fetch_mldata("MNIST Original")

# Extract the features and labels
    features = np.array(dataset.data, 'int16') 
    labels = np.array(dataset.target, 'int')

# Extract the hog features
    list_hog_fd = []
    for feature in features:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, 'float64')

# Normalize the features
    pp = preprocessing.StandardScaler().fit(hog_features)
    hog_features = pp.transform(hog_features)

    print( "Count of digits in dataset", Counter(labels))

# Create an linear SVM object
    clf = LinearSVC()

# Perform the training
    clf.fit(hog_features, labels)

# Save the classifier
    joblib.dump((clf, pp), "digits_cls.pkl", compress=3)
#////////////////////////////////////////

    return 0





def detectDigitImg():
    print("detImg")
#///////////////////////////////////////
# Get the path of the training set


# Load the classifier
    clf, pp = joblib.load("digits_cls.pkl")

# Read the input image 
    im = cv2.imread("photo_1.jpg")


# Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
    im2, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
    for rect in rects:
    # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
        nbr = clf.predict(roi_hog_fd)
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
        cv2.imshow("Resulting Image with Rectangular ROIs", im)
        cv2.waitKey()
#///////////////////////////////////////
    return 0





def detectDigitCam():
    print("detCam")
    return 0




def printMenu():
    print("\t//////////////////////////////////////\n\
           Hand Writing digits Recognition\n\
           \t 1. Segmentation and Train.\n\
           \t 2. Digits Detection with local image.\n\
           \t 3. Digits Detection with alive Camera.\n\
           \t 4. To be continue...\n\
           \t 5. Exit.\n\
       /////////////////////////////////////////\n")
    return 0

if __name__ =="__main__":
    printMenu();
    while(True):
        option = input("Your option(1-5): ")

        if(option=="1"):
            segmNtrain()
            printMenu()
        elif(option=="2"):       
            detectDigitImg()
            printMenu()
        elif(option=="3"):
            detectDigitCam()
            printMenu()
        elif(option=="4"):
            printMenu()
        elif(option=="5"):
            break




