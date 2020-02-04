import argparse
import open3d as o3d
import cv2
from skimage.io import imread, imsave
import numpy as np


class ViewerWithCallback:

    def __init__(self, config, device, align_depth_to_color):
        self.flag_exit = False
        self.align_depth_to_color = align_depth_to_color
        self.n_img = 0
        self.sensor = o3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(device):
            raise RuntimeError('Failed to connect to sensor')

    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def run(self):
        glfw_key_escape = 256
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.create_window('viewer', 1920, 540)
        print("Sensor initialized. Press [ESC] to exit.")

        vis_geometry_added = False
        maxx= 0
        while not self.flag_exit:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
            if rgbd is None:
                continue

            self.n_img += 1
            
            color_image = np.asanyarray(rgbd.color)
            depth_image = np.asanyarray(rgbd.depth) #*(256/14000)

            cv2.imwrite('./azure_data/depth/depth_'+str(self.n_img)+'.png', depth_image)
            cv2.imwrite('./azure_data/image/rgb_'+str(self.n_img)+'.png', color_image)
        
            # imsave('./azure_data/depth/depth_'+str(self.n_img)+'.png', depth_image)
            # imsave('./azure_data/image/rgb_'+str(self.n_img)+'.png', color_image)

            if not vis_geometry_added:
                vis.add_geometry(rgbd)
                vis_geometry_added = True

            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect mkv recorder.')
    parser.add_argument('--config', type=str, help='input json kinect config')
    parser.add_argument('--list',
                        action='store_true',
                        help='list available azure kinect sensors')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument('-a',
                        '--align_depth_to_color',
                        action='store_true',
                        help='enable align depth image to color')
    args = parser.parse_args()

    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    v = ViewerWithCallback(config, device, args.align_depth_to_color)
    v.run()
