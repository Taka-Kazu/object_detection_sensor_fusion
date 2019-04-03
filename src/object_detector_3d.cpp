#include <omp.h>

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>

#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>

#include "bounding_box_lib/bounding_box.h"

template<typename t_p>
class ObjectDetector3D
{
public:
    ObjectDetector3D(void);

    void callback(const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::PointCloud2ConstPtr&, const darknet_ros_msgs::BoundingBoxesConstPtr&);
    void sensor_fusion(const sensor_msgs::Image&, const sensor_msgs::CameraInfo&, const sensor_msgs::PointCloud2&, const darknet_ros_msgs::BoundingBoxes&);
    void get_color(double, int&, int&, int&);// dist, r, g, b
    void get_euclidean_cluster(pcl::PointCloud<t_p>&, pcl::PointCloud<t_p>&);// input, output
    int get_correspond_bb_index(cv::Point2d&, const darknet_ros_msgs::BoundingBoxes&);
    void get_color_bb(std::string, int&, int&, int&);
    double get_color_ratio(int, int, int);
    void coloring_pointcloud(pcl::PointCloud<t_p>&, int, int, int);

private:
    static constexpr double MAX_DISTANCE = 20.0;
    double LEAF_SIZE;
    double TOLERANCE;
    int MIN_CLUSTER_SIZE;
    int MAX_CLUSTER_SIZE;
    std::vector<std::string> CLASS_LABELS;
    int NUM_CLASS_LABELS;

    ros::NodeHandle nh;
    ros::NodeHandle private_nh;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::PointCloud2, darknet_ros_msgs::BoundingBoxes> sensor_fusion_sync_subs;
    message_filters::Subscriber<sensor_msgs::Image> image_sub;
    message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub;
    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> bb_sub;
    message_filters::Synchronizer<sensor_fusion_sync_subs> sensor_fusion_sync;

    ros::Publisher image_pub;
    ros::Publisher pc_pub;
    ros::Publisher semantic_cloud_pub;

    tf::TransformListener listener;
    tf::StampedTransform transform;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_detector_3d");
    ObjectDetector3D<pcl::PointXYZRGB> object_detector_3d;
    ros::spin();
    return 0;
}

template<typename t_p>
ObjectDetector3D<t_p>::ObjectDetector3D(void)
    : private_nh("~"),
      image_sub(nh, "/camera/color/image_raw", 10), camera_info_sub(nh, "/camera/color/camera_info", 10), pc_sub(nh, "/velodyne_points", 10), bb_sub(nh, "/darknet_ros/bounding_boxes", 10), sensor_fusion_sync(sensor_fusion_sync_subs(10), image_sub, camera_info_sub, pc_sub, bb_sub)
{
    image_pub = nh.advertise<sensor_msgs::Image>("/projection", 1);
    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/colored_cloud", 1);
    semantic_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud/colored/semantic", 1);
    sensor_fusion_sync.registerCallback(boost::bind(&ObjectDetector3D::callback, this, _1, _2, _3, _4));

    private_nh.param("LEAF_SIZE", LEAF_SIZE, 0.05);
    private_nh.param("TOLERANCE", TOLERANCE, 0.10);
    private_nh.param("MIN_CLUSTER_SIZE", MIN_CLUSTER_SIZE, 20);
    private_nh.param("MAX_CLUSTER_SIZE", MAX_CLUSTER_SIZE, 900);
    // for coloring
    nh.param("/darknet_ros/yolo_model/detection_classes/names", CLASS_LABELS, std::vector<std::string>(0));
    NUM_CLASS_LABELS = CLASS_LABELS.size();

    std::cout << "=== object_detector_3d ===" << std::endl;
    std::cout << "params: " << std::endl;
    std::cout << "LEAF_SIZE: " << LEAF_SIZE << std::endl;
    std::cout << "TOLERANCE: " << TOLERANCE << std::endl;
    std::cout << "MIN_CLUSTER_SIZE: " << MIN_CLUSTER_SIZE << std::endl;
    std::cout << "MAX_CLUSTER_SIZE: " << MAX_CLUSTER_SIZE << std::endl;
    std::cout << "NUM_CLASS_LABELS: " << NUM_CLASS_LABELS << std::endl;
}

template<typename t_p>
void ObjectDetector3D<t_p>::callback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::CameraInfoConstPtr& camera_info, const sensor_msgs::PointCloud2ConstPtr& pc, const darknet_ros_msgs::BoundingBoxesConstPtr& bbs)
{
    try{
        listener.waitForTransform(image->header.frame_id, pc->header.frame_id, ros::Time(0), ros::Duration(4.0));
        listener.lookupTransform(image->header.frame_id, pc->header.frame_id, ros::Time(0), transform);
        sensor_fusion(*image, *camera_info, *pc, *bbs);
    }catch(tf::TransformException ex){
        ROS_ERROR("%s", ex.what());
    }
}

template<typename t_p>
void ObjectDetector3D<t_p>::sensor_fusion(const sensor_msgs::Image& image, const sensor_msgs::CameraInfo& camera_info, const sensor_msgs::PointCloud2& pc, const darknet_ros_msgs::BoundingBoxes& bbs)
{
    double start_time = ros::Time::now().toSec();
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(pc, *lidar_cloud);

    typename pcl::PointCloud<t_p>::Ptr cloud(new pcl::PointCloud<t_p>);
    pcl::copyPointCloud(*lidar_cloud, *cloud);

    typename pcl::PointCloud<t_p>::Ptr trans_cloud(new pcl::PointCloud<t_p>);
    pcl_ros::transformPointCloud(*cloud, *trans_cloud, transform);

    cv_bridge::CvImageConstPtr cv_img_ptr;
    try{
        cv_img_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
    }catch(cv_bridge::Exception& ex){
        ROS_ERROR("cv_bridge exception: %s", ex.what());
        return;
    }

    cv::Mat cv_image(cv_img_ptr->image.rows, cv_img_ptr->image.cols, cv_img_ptr->image.type());
    cv_image = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8)->image;

    cv::Mat rgb_image;
    cv::cvtColor(cv_image, rgb_image, CV_BGR2RGB);

    image_geometry::PinholeCameraModel cam_model;
    cam_model.fromCameraInfo(camera_info);

    typename pcl::PointCloud<t_p>::Ptr colored_cloud(new pcl::PointCloud<t_p>);
    *colored_cloud = *trans_cloud;
    cv::Mat projection_image = rgb_image.clone();

    int n_bbs = bbs.bounding_boxes.size();
    std::vector<typename pcl::PointCloud<t_p>::Ptr> bb_clouds;
    for(size_t i=0;i<n_bbs;i++){
        typename pcl::PointCloud<t_p>::Ptr _pc(new pcl::PointCloud<t_p>);
        bb_clouds.push_back(_pc);
    }

    for(auto& pt : colored_cloud->points){
        if(pt.z<0){
            // behind camera
            pt.b = 255;
            pt.g = 255;
            pt.r = 255;
        }else{
            cv::Point3d pt_cv(pt.x, pt.y, pt.z);
            cv::Point2d uv;
            uv = cam_model.project3dToPixel(pt_cv);

            if(uv.x > 0 && uv. x < rgb_image.cols && uv.y > 0 && uv.y < rgb_image.rows){
                pt.b = rgb_image.at<cv::Vec3b>(uv)[0];
                pt.g = rgb_image.at<cv::Vec3b>(uv)[1];
                pt.r = rgb_image.at<cv::Vec3b>(uv)[2];

                double distance = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
                int r, g, b;
                get_color(distance, r, g, b);
                cv::circle(projection_image, uv, 1, cv::Scalar(b, g, r), -1);

                // in bounding box
                int bb_index = get_correspond_bb_index(uv, bbs);
                if(bb_index >= 0){
                    bb_clouds[bb_index]->points.push_back(pt);
                }
            }else{
                pt.b = 255;
                pt.g = 255;
                pt.r = 255;
            }
        }
    }

    // semantic cloud
    typename pcl::PointCloud<t_p>::Ptr semantic_cloud(new pcl::PointCloud<t_p>);
    for(int i=0;i<n_bbs;i++){
        std::cout << "label: " << bbs.bounding_boxes[i].Class << std::endl;
        int r, g, b;
        get_color_bb(bbs.bounding_boxes[i].Class, r, g, b);
        coloring_pointcloud(*(bb_clouds[i]), r, g, b);
        typename pcl::PointCloud<t_p>::Ptr cluster(new pcl::PointCloud<t_p>);
        get_euclidean_cluster(*(bb_clouds[i]), *cluster);
        *semantic_cloud += *cluster;
    }

    pcl_ros::transformPointCloud(*semantic_cloud, *semantic_cloud, transform.inverse());
    sensor_msgs::PointCloud2 output_semantic_cloud;
    pcl::toROSMsg(*semantic_cloud, output_semantic_cloud);
    output_semantic_cloud.header = pc.header;
    semantic_cloud_pub.publish(output_semantic_cloud);

    // colored cloud
    typename pcl::PointCloud<t_p>::Ptr output_cloud(new pcl::PointCloud<t_p>);
    pcl_ros::transformPointCloud(*colored_cloud, *output_cloud, transform.inverse());

    sensor_msgs::PointCloud2 output_pc;
    pcl::toROSMsg(*output_cloud, output_pc);
    output_pc.header.frame_id = pc.header.frame_id;
    output_pc.header.stamp = ros::Time::now();
    pc_pub.publish(output_pc);

    // projection image
    sensor_msgs::ImagePtr output_image;
    output_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", projection_image).toImageMsg();
    output_image->header.frame_id = image.header.frame_id;
    output_image->header.stamp = ros::Time::now();
    image_pub.publish(output_image);

    std::cout << ros::Time::now().toSec() - start_time << "[s]" << std::endl;
}

template<typename t_p>
void ObjectDetector3D<t_p>::get_color(double d, int &r, int &g, int &b)
{
    r = 255;
    g = 255;
    b = 255;

    if(d < 0){
        d = 0;
    }else if(d > MAX_DISTANCE){
        d = MAX_DISTANCE;
    }

    double v = d / MAX_DISTANCE * 255;// 0 ~ 255

    if(v < (0.25 * 255)){
        r = 0;
        g = 4 * v;
    }else if(v < 0.5 * 255){
        r = 0;
        b = 255 + 4 * (0.25 * 255 - v);
    }else if(v < 0.75 * 255){
        r = 4 * (v - 0.5 * 255);
        b = 0;
    }else{
        g = 255 + 4 * (0.75 * 255 - v);
        b = 0;
    }
}

template<typename t_p>
void ObjectDetector3D<t_p>::get_euclidean_cluster(pcl::PointCloud<t_p>& pc, pcl::PointCloud<t_p>& output_pc)
{
    typename pcl::PointCloud<t_p>::Ptr cloud_filtered(new pcl::PointCloud<t_p>);
    pcl::copyPointCloud(pc, *cloud_filtered);

    typename pcl::VoxelGrid<t_p> vg;
    vg.setInputCloud(cloud_filtered);
    vg.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
    vg.filter(*cloud_filtered);

    // z to 0
    size_t n_points = cloud_filtered->points.size();
    std::vector<double> original_z(n_points);
    for(size_t i=0;i<n_points;++i){
        original_z[i] = cloud_filtered->points[i].z;
        cloud_filtered->points[i].z = 0;
    }

    typename pcl::search::KdTree<t_p>::Ptr tree(new pcl::search::KdTree<t_p>);
    tree->setInputCloud(cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    typename pcl::EuclideanClusterExtraction<t_p> ec;
    ec.setClusterTolerance(TOLERANCE);
    ec.setMinClusterSize(MIN_CLUSTER_SIZE);
    ec.setMaxClusterSize(MAX_CLUSTER_SIZE);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);

    // restore to original z
    for(size_t i=0;i<n_points;++i){
        cloud_filtered->points[i].z = original_z[i];
    }
    if(!cluster_indices.size() > 0){
        std::cout << "!!! clustering error !!!" << std::endl;
        return;
    }

    int max_cluster_size = 0;
    int max_cluster_index = -1;
    int index = 0;
    for(auto indices : cluster_indices){
        typename pcl::PointCloud<t_p>::Ptr cluster(new pcl::PointCloud<t_p>);
        for(auto it : indices.indices){
            cluster->points.push_back(cloud_filtered->points.at(it));
        }
        int cluster_size = cluster->points.size();
        if(cluster_size > max_cluster_size){
            max_cluster_index = index;
            max_cluster_size = cluster_size;
            output_pc = *cluster;
        }
        index++;
    }
    std::cout << "cluster size: " << output_pc.points.size() << std::endl;
}

template<typename t_p>
int ObjectDetector3D<t_p>::get_correspond_bb_index(cv::Point2d& uv, const darknet_ros_msgs::BoundingBoxes& bbs)
{
    int counter = 0;
    for(auto bb : bbs.bounding_boxes){
        if(bb.xmin <= uv.x && uv.x <= bb.xmax && bb.ymin <= uv.y && uv.y <= bb.ymax){
           return counter;
        }
        counter++;
    }
    // return -1 if the point is not in any bounding box
    return -1;
}

template<typename t_p>
void ObjectDetector3D<t_p>::get_color_bb(std::string label, int& r, int& g, int& b)
{
    auto it = std::find(CLASS_LABELS.begin(), CLASS_LABELS.end(), label);
    int index = std::distance(CLASS_LABELS.begin(), it);
    int offset = index * 123457 % NUM_CLASS_LABELS;
    r = 255 * get_color_ratio(2, offset, NUM_CLASS_LABELS);
    g = 255 * get_color_ratio(1, offset, NUM_CLASS_LABELS);
    b = 255 * get_color_ratio(0, offset, NUM_CLASS_LABELS);
}

template<typename t_p>
double ObjectDetector3D<t_p>::get_color_ratio(int color_num, int value, int max)
{
    // color_num -> r:2 g:1 b:0
    double colors[6][3] = {{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
    double ratio = ((double)value / max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    ratio = (1-ratio) * colors[i][color_num] + ratio * colors[j][color_num];
    return ratio;
}

template<typename t_p>
void ObjectDetector3D<t_p>::coloring_pointcloud(pcl::PointCloud<t_p>& cloud, int r, int g, int b)
{
    for(auto it=cloud.points.begin();it!=cloud.points.end();++it){
        it->r = r;
        it->g = g;
        it->b = b;
    }
}
