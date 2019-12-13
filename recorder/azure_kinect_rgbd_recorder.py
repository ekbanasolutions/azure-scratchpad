import argparse
import open3d as o3d
import numpy as np
# import scipy.misc
import cv2
from skimage.io import imread, imsave
import numpy as np

class ViewerWithCallback:

    def __init__(self, config, device, align_depth_to_color):
        self.flag_exit = False
        self.align_depth_to_color = align_depth_to_color

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
        n_img = 0
        while not self.flag_exit:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
            if rgbd is None:
                continue

            n_img += 1
            color_image = np.asanyarray(rgbd.color)
            depth_image = np.asanyarray(rgbd.depth)
            imsave('./azure_data/depth/depth_'+str(n_img)+'.png', depth_image)
            imsave('./azure_data/image/rgb_'+str(n_img)+'.png', color_image)

            if not vis_geometry_added:
                vis.add_geometry(rgbd)
                vis_geometry_added = True

            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()


class PCDViewerWithCallback:

    def __init__(self, config, device, align_depth_to_color = True):
        self.flag_exit = False
        self.align_depth_to_color = align_depth_to_color

        self.sensor = o3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(device):
            raise RuntimeError('Failed to connect to sensor')

    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def run_pointcloud(self):
        glfw_key_escape = 256
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.create_window('viewer', 1920, 540)
        
        print("This is for points cloud visualization. Press [ESC] to exit.")

        vis_geometry_added = False
        pcd = None
        n_img = 0;
        while not self.flag_exit:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
            if rgbd is None:
                continue
                
            n_img += 1
            # my_camera_intrinsics = o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            my_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
            my_camera_intrinsics.set_intrinsics(1280, 720, 601.71393, 601.49011, 636.46729, 366.23587)

            # I need to do the following because of some apparent API error
            # the rgbd image formed by rgbd = self.sensor.capture_frame() above doesn't
            # seem to comply with pcd = o3d.geometry.PointCloud.create_from_rgbd_image() below
            # so i need to recreate the rgbd image using it's own data!!
            # funny, but works
            color_image = np.asanyarray(rgbd.color)
            depth_image = np.asanyarray(rgbd.depth)

            # scipy.misc.imsave('./azure_data/rgb/rgb_'+str(n_img)+'.jpg', rgb)
            # scipy.misc.imsave('./azure_data/depth/depth_'+str(n_img)+'.jpg', depth)

            x = np.ones((100, 100), dtype=np.uint16)
            imsave('./azure_data/depth/depth_'+str(n_img)+'.png', depth_image)
            imsave('./azure_data/image/rgb_'+str(n_img)+'.png', color_image)


            new_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgbd.color, rgbd.depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

            is_success, motion_matrix, info_matrix = o3d.odometry.compute_rgbd_odometry(rgbd, rgbd, pinhole_camera_intrinsic=my_camera_intrinsics)
            rgbd = new_rgbd

            if pcd != None:
                # comment the code below if only instantaneous pcd is to be visualized
                vis.remove_geometry(pcd)
                pass

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, my_camera_intrinsics, extrinsic=motion_matrix)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            # if not vis_geometry_added:
            # vis.add_geometry(pcd)
            vis_geometry_added = False

            if not vis_geometry_added:
                vis.add_geometry(pcd)
                vis_geometry_added = True

            # vis.update_geometry()
            vis.poll_events()
            # vis.update_renderer()

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
        print (config)

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    align_depth_to_color = args.align_depth_to_color

    # v = PCDViewerWithCallback(config, device)
    # v.run_pointcloud()

    v = ViewerWithCallback(config, device, align_depth_to_color)
    v.run()
    