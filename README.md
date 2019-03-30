# object_detection_sensor_fusion

## Environment
- Ubuntu16.04
- ROS(kinetic)
- OpenCV(>=3.3)
- PCL(>=1.7)
- darknet_ros(https://github.com/leggedrobotics/darknet_ros)

## Node
### object_detector_3d
#### published topics
- /projection (sensor_msgs/Image)
- /colored_cloud (sensor_msgs/PointCloud2)
#### subscribed topics
- /camera/color/image_raw (sensor_msgs/Image)
- /camera/color/camera_info (sensor_msgs/CameraInfo)
- /velodyne_points (sensor_msgs/PointCloud2)
- /darknet_ros/bounding_boxes (darknet_ros_msgs/BoundingBoxes)
