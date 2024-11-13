from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import pyk4a


import numpy as np
import cv2
import torch
from pyk4a import CalibrationType



class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass




class KinectDataset2(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        
        # Configure the Kinect camera settings
        self.kinect = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_1080P,
                depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                synchronized_images_only=True,
            )
        )
        self.kinect.start()
        self.kinect.whitebalance = 4500
        assert self.kinect.whitebalance == 4500
        self.kinect.whitebalance = 4510
        assert self.kinect.whitebalance == 4510
        # Retrieve and set intrinsic parameters
        calib  = self.kinect.calibration
        #print(dir(calib))
    # Get the color camera intrinsic parameters
        color_camera_matrix = calib.get_camera_matrix(CalibrationType.COLOR)
        self.fx = color_camera_matrix[0, 0]
        self.fy = color_camera_matrix[1, 1]
        self.cx = color_camera_matrix[0, 2]
        self.cy = color_camera_matrix[1, 2]
        color_res = calib.color_resolution
        if color_res == ColorResolution.RES_720P:
            self.width = 1280
            self.height = 720
        elif color_res == ColorResolution.RES_1080P:
            self.width = 1920
            self.height = 1080
        elif color_res == ColorResolution.RES_3072P:
            self.width = 3840
            self.height = 2160
        else:
            # Handle other resolutions or set default values
            self.width = 640  # Default width
            self.height = 480  # Default height
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        self.has_depth = config["Dataset"]["sensor_type"] == "depth"

        # FOV calculations (optional, using a similar approach as before)
        self.fovx = np.degrees(2 * np.arctan(self.width / (2 * self.fx)))
        self.fovy = np.degrees(2 * np.arctan(self.height / (2 * self.fy)))
        
        # Handle image undistortion if necessary
        self.dist_coeffs = calib.get_distortion_coefficients(CalibrationType.COLOR)
        if self.dist_coeffs.any():
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                self.K, self.dist_coeffs, np.eye(3), self.K, (self.width, self.height), cv2.CV_32FC1
            )

        
            
    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        frame, image, depth = None, None, None
        
        capture = self.kinect.get_capture()
        

        if capture.transformed_depth is not None:
            print("transformed_Depth is available")
            #cv2.imshow("Transformed Depth", colorize(capture.transformed_depth, (None, 5000)))
            depth_scale = self.kinect.calibration.depth_mode.value  # Use the depth mode value instead
            # Apply the depth scale
            depth = capture.transformed_depth.astype(np.float32) * depth_scale
            depth[depth < 0] = 0
            np.nan_to_num(depth, nan=1000)
        else:
            print("transformed_Depth is not available")
            depth = None
            frame = capture.color
        
        frame = capture.color
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        if self.dist_coeffs.any():
            frame = cv2.remap(frame, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(frame / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        return image, depth, pose



class KinectDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        
        # Configure the Kinect camera settings
        self.kinect = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_2160P,
                depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
                synchronized_images_only=True,
            )
        )
        self.kinect.start()
        self.kinect.whitebalance = 4500
        assert self.kinect.whitebalance == 4500
        self.kinect.whitebalance = 4510
        assert self.kinect.whitebalance == 4510
        # Retrieve and set intrinsic parameters
        calib  = self.kinect.calibration
        #print(dir(calib))
    # Get the color camera intrinsic parameters
        color_camera_matrix = calib.get_camera_matrix(CalibrationType.COLOR)
        self.fx = color_camera_matrix[0, 0]
        self.fy = color_camera_matrix[1, 1]
        self.cx = color_camera_matrix[0, 2]
        self.cy = color_camera_matrix[1, 2]
        color_res = calib.color_resolution
        if color_res == ColorResolution.RES_720P:
            self.width = 1280
            self.height = 720
        elif color_res == ColorResolution.RES_1080P:
            self.width = 1920
            self.height = 1080
        elif color_res == ColorResolution.RES_3072P:
            self.width = 3840
            self.height = 2160
        else:
            # Handle other resolutions or set default values
            self.width = 640  # Default width
            self.height = 480  # Default height
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        self.has_depth = config["Dataset"]["sensor_type"] == "depth"

        # FOV calculations (optional, using a similar approach as before)
        self.fovx = np.degrees(2 * np.arctan(self.width / (2 * self.fx)))
        self.fovy = np.degrees(2 * np.arctan(self.height / (2 * self.fy)))
        
        # Handle image undistortion if necessary
        self.dist_coeffs = calib.get_distortion_coefficients(CalibrationType.COLOR)
        if self.dist_coeffs.any():
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                self.K, self.dist_coeffs, np.eye(3), self.K, (self.width, self.height), cv2.CV_32FC1
            )

        
            
    # def __getitem__(self, idx):
    #     pose = torch.eye(4, device=self.device, dtype=self.dtype)
    #     image, depth = None, None
        
    #     try:
    #         capture = self.kinect.get_capture()
            
    #         # Retrieve color frame
    #         if capture.color is not None:
    #             frame = capture.color
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    #             if self.dist_coeffs.any():
    #                 frame = cv2.remap(frame, self.map1x, self.map1y, cv2.INTER_LINEAR)

    #             image = (
    #                 torch.from_numpy(frame / 255.0)
    #                 .clamp(0.0, 1.0)
    #                 .permute(2, 0, 1)
    #                 .to(device=self.device, dtype=self.dtype)
    #             )

    #         # Retrieve depth frame if available
    #         # if self.has_depth and capture.depth is not None:
    #         #     #depth_scale = self.kinect.calibration.get_depth_scale  # Remove the parentheses
    #         #     depth_scale = self.kinect.calibration.depth_mode.value  # Use the depth mode value instead
    #         if capture.transformed_depth is not None:
    #             #cv2.imshow("Transformed Depth", colorize(capture.transformed_depth, (None, 5000)))
    #             depth_scale = self.kinect.calibration.depth_mode.value  # Use the depth mode value instead
    #             # Apply the depth scale
    #             depth = capture.transformed_depth.astype(np.float32) * depth_scale
    #             depth[depth < 0] = 0
    #             np.nan_to_num(depth, nan=1000)

    #         return image, depth, pose

    #     except Exception as e:
    #         print(f"Error capturing data from Kinect: {e}")
    #         return image, depth, pose

    # def __del__(self):
    #     if hasattr(self, 'kinect'):
    #         self.kinect.stop()

    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        image, depth = None, None

        try:
            capture = self.kinect.get_capture()
            
            # Retrieve color frame
            if capture.color is not None:
                frame = capture.color
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                if self.dist_coeffs.any():
                    frame = cv2.remap(frame, self.map1x, self.map1y, cv2.INTER_LINEAR)
                image = (
                    torch.from_numpy(frame / 255.0)
                    .clamp(0.0, 1.0)
                    .permute(2, 0, 1)
                    .to(device=self.device, dtype=self.dtype)
                )

            # Retrieve and process depth frame if available
            if self.has_depth and capture.transformed_depth is not None:
                depth_scale = self.kinect.calibration.depth_mode.value  # Use the depth mode value instead
                depth = capture.transformed_depth.astype(np.float32) * depth_scale
                depth[depth < 0] = 0
                np.nan_to_num(depth, nan=1000)

        except Exception as e:
            print(f"Error capturing data from Kinect: {e}")

        return image, depth, pose



def main():
    # Arguments and config can be placeholders if not used in the dataset initialization.
    args = None
    path = "output"
    config = {"Dataset": {"sensor_type": "depth"}}  # Change "depth" to "color" if you only want RGB
    
    # Initialize the Kinect dataset
    dataset = KinectDataset2(args, path, config)
    
    # Set the number of frames to capture for testing
    num_frames_to_capture = 100
    
    for idx in range(num_frames_to_capture):
        # Get image, depth, and pose from the dataset
        image, depth, pose = dataset[idx]
        
        # Convert image tensor back to a format OpenCV can display
        if image is not None:
            image_np = image.permute(1, 2, 0).cpu().numpy() * 255
            image_np = image_np.astype(np.uint8)

            # Display the image
            cv2.imshow("RGB Image", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # Display the depth map if available
        if depth is not None:
            depth_np = depth
            #depth_np = depth.cpu().numpy()
            depth_display = (depth_np / depth_np.max() * 255).astype(np.uint8)  # Normalize for display
            cv2.imshow("Depth Image", depth_display)

        # Print the pose matrix
        print("Pose Matrix:\n", pose)

        # Save the frames as images for verification
        cv2.imwrite(f"{path}/rgb_frame_{idx}.png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        if depth is not None:
            cv2.imwrite(f"{path}/depth_frame_{idx}.png", depth_display)

        # Wait briefly, exit on 'q' press
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    del dataset  # This will stop the Kinect camera

if __name__ == "__main__":
    main()
