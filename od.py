'''

File name: od.py 
Description: This script performs object detection.

Author(s):  Victor Santiago Solis Garcia
            Jonathan Ariel Valadez Saldaña 

Creation date: 03/22/2024

Usage example:  python3 test-object-detection.py --video_file football-field-cropped-video.mp4 
--frame_resize_percentage 30

'''


import cv2 
import argparse
import numpy as np
from numpy.typing import NDArray

def parser_user_data()->argparse:
    '''
    Function to receive the user data 
    Parameters:    None
    Returns:       args(argparse): argparse object with the video path and the frame resize object
    '''

    parser = argparse.ArgumentParser(description='Tunning HSV bands for object detection')
    parser.add_argument('--video_file', 
                        type=str, 
                        default='camera', 
                        help='Video file used for the object detection process')
    parser.add_argument('--frame_resize_percentage', 
                        type=int, 
                        help='Rescale the video frames, e.g., 20 if scaled to 20%')
    args = parser.parse_args()

    return args

def rescale_frame(frame:NDArray, percentage:np.intc=5)->NDArray:
    '''
    Function to rescale the frame to an specific porcentage
    Parameters:    frame(NDArray): The input image frame, this is a NumPy array 
                   percentage(intc): Rescale percentage, the default value is 5 wich means     
                                       the output frame will b 5% of the original value
    Returns:       frame(NDArray): The resized frame
    '''
    width = int(frame.shape[1] * percentage / 100)
    height = int(frame.shape[0] * percentage / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame

def adjust_hsv_threshold(frame:NDArray, sensitivity:np.intc=50)-> NDArray:
    '''
    Function to rescale the frame to an specific porcentage
    Parameters:    frame(NDArray):  The input image frame, this is a NumPy array 
                   sensitivity(intc): Factor to adjust the hsv based on the median
    Returns:       mask(NDArray): hsv mask
    '''

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    median_v = np.median(hsv[:,:,2])
    
    lower_hsv = np.array([max(0, median_v - sensitivity), 0, 0],dtype=np.uint8)
    upper_hsv = np.array([179, 200, min(255, median_v + sensitivity)],dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask

def segment_object(args:argparse) -> None:
    '''
    Function to rescale the frame to an specific porcentage
    Parameters:    args(argparse):  Argparse object with the user info.
                        args.video_file(str): path to the video
                        args.frame_resize_percentage(int): rescale percentage
                   sensitivity(intc): Factor to adjust the hsv based on the median
    Returns:       None
    '''
    cap = cv2.VideoCapture(args.video_file)

    if not cap.isOpened():
        print("Can't open the file")
        return
    roi = None
    no_contour_detected_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("The video ended or the frame could not be readed")
                break

            frame = rescale_frame(frame, args.frame_resize_percentage)

            if roi and no_contour_detected_count < 10: 
                x, y, w, h = roi
                frame_roi = frame[y:y+h, x:x+w]
            else:
                frame_roi = frame
                roi = None 

            mask = adjust_hsv_threshold(frame_roi)
            hsv_filtered_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask)

            gray = cv2.cvtColor(hsv_filtered_frame, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                if roi:
                    x += roi[0]
                    y += roi[1]

                margin = 50
                roi = [max(x - margin, 0), max(y - margin, 0), w + 2 * margin, h + 2 * margin]
                no_contour_detected_count = 0  
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
            else:
                no_contour_detected_count += 1

            cv2.imshow('Frame', frame)
            #cv2.imshow('ROI',gray)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def run_pipeline()->None:
    args = parser_user_data()
    segment_object(args)