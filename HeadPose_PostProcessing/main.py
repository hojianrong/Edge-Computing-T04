import argparse
import os
import cv2
import json
import time
import numpy as np
import select
import sys
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

"""
If Pose_x is less than -10 = "Looking Downards"
Otherwise Pose is "Looking Foward" 
If Pose_y is more than 15 =  "Looking Right"
If Pose_y is less than -15 = "Looking Left" 
"""

class FaceInference:
    def __init__(self):
        # Setup a face detector to detect human faces.
        self.face_detector = FaceDetector("assets/face_detector.onnx")

        # Setup the mark detector to detect landmarks.
        self.mark_detector = MarkDetector("assets/face_landmarks.onnx")
    
        # Setup the pose estimator to solve pose.
        self.pose_estimator = None
        
    def LoadPoseEstimator(self, frame_width, frame_height):
        self.pose_estimator = PoseEstimator(frame_width, frame_height)
        
        
    def Preprocess_Store_Image(self, save_dir, imgs_dir, capture_interval):
        # Image input size is 640 x 480
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
    
        for img in os.listdir(imgs_dir): # Delete all files in images folder if exist
            if os.path.isfile(os.path.join(imgs_dir, img)):
                os.remove(os.path.join(imgs_dir, img))
                
        for img in os.listdir(save_dir): # Delete all files in results folder if exist
            if os.path.isfile(os.path.join(save_dir, img)):
                os.remove(os.path.join(save_dir, img))


        cap = cv2.VideoCapture(0)  # Capture from camera
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # print("Frame Width", frame_width)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print("Frame Height", frame_height)
        frame_count = 0 # Frame Counter
        
        start_time = time.time() # Start time of program

        while True:
            ret, orig_image = cap.read()
            if not ret:
                print("Error occurred, could not capture any frames")
                break

            # Save frames for later inference
            # Take the current time subtract the time the capturing started
            elapsed_time = time.time() - start_time
            if elapsed_time >= capture_interval: # Capture a frame every 0.1 seconds (10 FPS)
                start_time = time.time() # Reset the starting timer
                frame_count += 1 # Count frames
                capture_time = int(time.time()) # Obtain the time of capture
                
                # Saving of original image for inference 
                # Saved in Images folder
                frame_path = os.path.join(imgs_dir, f"frame_{frame_count}_{capture_time}.jpg")
                cv2.imwrite(frame_path, orig_image)
                print(f"Frame saved at: {frame_path}")

            #cv2.imshow('frame', orig_image) # Uncomment this line to show the Opencv Window
            
            # Press q once done to exit
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key.lower() == 'q':
                    print("Exiting...")
                    break

        cap.release()
        cv2.destroyAllWindows()
        
        return frame_width, frame_height

    def Inference(self, imgs_dir, save_dir):
        # Perform inference on saved frames
        # Dictionary to save the timestamp / frame number & bounding box coordinates
        image_data = {}

        # Loop through each file in folder
        for img_name in os.listdir(imgs_dir):
            img_path = os.path.join(imgs_dir, img_name)
    
            # Read each image with cv2
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            # Step 1: Get faces from current frame.
            # Measure the time before processing the frame
            frame_start_time = time.time()
            
            faces, _ = self.face_detector.detect(image, 0.7)
            
            # Any valid face found?
            if len(faces) > 0:
                face_coordinates = []
                for face in faces:
                    # Step 2: Detect landmarks. Crop and feed the face area into the
                    # mark detector. Note only the first face will be used for
                    # demonstration.
                    x1, y1, x2, y2 = face[:4].astype(int)
                    patch = image[y1:y2, x1:x2]

                    # Run the mark detection.
                    marks = self.mark_detector.detect([patch])[0].reshape([68, 2])

                    # Convert the locations from local face area to the global image.
                    marks *= (x2 - x1)
                    marks[:, 0] += x1
                    marks[:, 1] += y1
                
                    # Step 3: Try pose estimation with 68 points.
                    pose = self.pose_estimator.solve(marks)
                
                    # Extract rotation vector x and y components
                    rvec = pose[0]
               
                    #rotation_x = pose[0][0][0]  # Extracting x component from the first array
                    #rotation_y = pose[0][1][0]  # Extracting y component from the first array
                    #rotation_z = pose[0][2][0]  # Extracting z component from the first array
                
                    # Call the function to extract Euler angles from rotation vectors
                    rot_params = self.pose_estimator.rot_params_rv(rvec)
                
                    rot_extract = rot_params[:2]
                
                    # Append to list
                    face_coordinates.append(rot_extract)
                
                    ## Add to dictionary
                    image_data[img_name] = face_coordinates  # Store only essential rotation angles
                
                    # Visualize the estimated pose on the image
                    self.pose_estimator.visualize(image, pose)
                
                    # Measure the time after processing the frame
                    frame_end_time = time.time()
                
                    # Calculate and print the inference time for this frame
                    inference_time = frame_end_time - frame_start_time
                    print(f"Inference time: {inference_time:.6f} seconds")
                
                    # Save the new predicted image 
                    cv2.imwrite(os.path.join(save_dir, f"{img_name}_predicted.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        print("Inference on saved frames completed.")

        # Save the image data dictionary to a JSON file
        json_file_path = os.path.join(save_dir, "bounding_box_data.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(image_data, json_file)
        
        print("Saving to JSON format completed.")


def main():
    save_dir = "Results"  # Directory to save the detected faces and bounding boxes 
    imgs_dir = "Images"  # Directory containing images for inference
    # Get the frame size. This will be used by the following detectors.
    capture_interval = 0.1 # Estimated 10 FPS

    
    face_inference = FaceInference()
    frame_width, frame_height = face_inference.Preprocess_Store_Image(save_dir, imgs_dir, capture_interval)
    face_inference.LoadPoseEstimator(frame_width, frame_height)
    face_inference.Inference(imgs_dir, save_dir)
    
    

if __name__ == "__main__":
    main()
