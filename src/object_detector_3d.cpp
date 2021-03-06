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
    void get_euclidean_cluster(typename pcl::PointCloud<t_p>::Ptr&, pcl::PointCloud<t_p>&);// input, output
    int get_correspond_bb_index(cv::Point2d&, const darknet_ros_msgs::BoundingBoxes&);
    void get_color_bb(std::string, int&, int&, int&);
    double get_color_ratio(int, int, int);
    void coloring_pointcloud(typename pcl::PointCloud<t_p>::Ptr&, int, int, int);
    void get_closest_point(typename pcl::PointCloud<t_p>::Ptr&, t_p&);
    void principal_component_analysis(typename pcl::PointCloud<t_p>::Ptr&, double&);

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
    ros::Publisher projection_semantic_image_pub;
    ros::Publisher bb_pub;

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
    projection_semantic_image_pub = nh.advertise<sensor_msgs::Image>("/projection/semantic", 1);
    bb_pub = nh.advertise<visualization_msgs::MarkerArray>("/bounding_boxes_3d", 1);
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

    cv::Mat bgr_image = cv_image.clone();

    image_geometry::PinholeCameraModel cam_model;
    cam_model.fromCameraInfo(camera_info);

    typename pcl::PointCloud<t_p>::Ptr colored_cloud(new pcl::PointCloud<t_p>);
    *colored_cloud = *trans_cloud;
    cv::Mat projection_image = bgr_image.clone();

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

            if(uv.x > 0 && uv. x < bgr_image.cols && uv.y > 0 && uv.y < bgr_image.rows){
                pt.b = bgr_image.at<cv::Vec3b>(uv)[0];
                pt.g = bgr_image.at<cv::Vec3b>(uv)[1];
                pt.r = bgr_image.at<cv::Vec3b>(uv)[2];

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
        coloring_pointcloud(bb_clouds[i], r, g, b);
        typename pcl::PointCloud<t_p>::Ptr cluster(new pcl::PointCloud<t_p>);
        get_euclidean_cluster(bb_clouds[i], *cluster);
        pcl::copyPointCloud(*cluster, *bb_clouds[i]);
        *semantic_cloud += *bb_clouds[i];
    }

    typename pcl::PointCloud<t_p>::Ptr output_sc(new pcl::PointCloud<t_p>);
    pcl::copyPointCloud(*semantic_cloud, *output_sc);
    pcl_ros::transformPointCloud(*output_sc, *output_sc, transform.inverse());
    sensor_msgs::PointCloud2 output_semantic_cloud;
    pcl::toROSMsg(*output_sc, output_semantic_cloud);
    output_semantic_cloud.header = pc.header;
    semantic_cloud_pub.publish(output_semantic_cloud);

    // projection/semantic image
    cv::Mat projection_semantic_image;
    projection_image.copyTo(projection_semantic_image);
    for(int i=0;i<n_bbs;i++){
        auto bb = bbs.bounding_boxes[i];
        int r, g, b;
        get_color_bb(bb.Class, r, g, b);
        cv::rectangle(projection_semantic_image, cv::Point(bb.xmin, bb.ymin), cv::Point(bb.xmax, bb.ymax), cv::Scalar(b, g, r), 3);
    }
    for(auto& pt : semantic_cloud->points){
        cv::Point3d pt_cv(pt.x, pt.y, pt.z);
        cv::Point2d uv;
        uv = cam_model.project3dToPixel(pt_cv);
        cv::circle(projection_semantic_image, uv, 3, cv::Scalar(pt.b, pt.g, pt.r), -1);
    }
    sensor_msgs::ImagePtr output_psi;
    output_psi = cv_bridge::CvImage(std_msgs::Header(), "bgr8", projection_semantic_image).toImageMsg();
    output_psi->header.frame_id = image.header.frame_id;
    output_psi->header.stamp = ros::Time::now();
    projection_semantic_image_pub.publish(output_psi);

    //std::cout << "=== bb ===" << std::endl;
    // generate bounding box
    visualization_msgs::MarkerArray bbs_3d;
    for(int i=0;i<n_bbs;i++){
        if(bb_clouds[i]->points.size() > 0){
            //std::cout << "label: " << bbs.bounding_boxes[i].Class << std::endl;
            bounding_box_lib::BoundingBox bb;
            cv::Point2d uv[4];
            cv::Point3d corner_vec[4];
            double vector_ratio[4];
            long int x_limit[2] = {bbs.bounding_boxes[i].xmin, bbs.bounding_boxes[i].xmax};
            long int y_limit[2] = {bbs.bounding_boxes[i].ymin, bbs.bounding_boxes[i].ymax};
            t_p closest_point;
            get_closest_point(bb_clouds[i], closest_point);
            for(int j=0;j<4;j++){
                uv[j].x = x_limit[(j<2) ? 0 : 1];
                uv[j].y = y_limit[(j%2==0) ? 0 : 1];
                corner_vec[j] = cam_model.projectPixelTo3dRay(uv[j]);
                vector_ratio[j] = closest_point.z / corner_vec[j].z;
                corner_vec[j].x *= vector_ratio[j];
                corner_vec[j].y *= vector_ratio[j];
                corner_vec[j].z *= vector_ratio[j];
                //std::cout << corner_vec[j] << std::endl;
            }
            double width = corner_vec[2].x - corner_vec[0].x;
            double height = corner_vec[1].y - corner_vec[0].y;
            double depth = width;
            bb.set_id(i);
            bb.set_frame_id(image.header.frame_id);

            std::cout << bbs.bounding_boxes[i].Class << std::endl;
            double yaw;
            principal_component_analysis(bb_clouds[i], yaw);

            bb.set_orientation(0, -yaw, 0);
            bb.set_scale(width, height, depth);
            int r, g, b;
            get_color_bb(bbs.bounding_boxes[i].Class, r, g, b);
            bb.set_rgb(r, g, b);
            bb.set_centroid(corner_vec[0].x + width * 0.5, corner_vec[0].y + height * 0.5, corner_vec[0].z + depth * 0.5);
            //std::cout << width << ", " << height << ", " << depth << std::endl;
            //std::cout << corner_vec[0].x + width * 0.5 << ", " << corner_vec[0].y + height * 0.5 << ", " << corner_vec[0].z + depth * 0.5 << std::endl;
            bb.calculate_vertices();
            bbs_3d.markers.push_back(bb.get_bounding_box());
        }
    }
    static int last_n_bbs = 0;
    for(int i=n_bbs;i<last_n_bbs;i++){
        visualization_msgs::Marker m;
        m.action = visualization_msgs::Marker::DELETE;
        m.id = i;
        m.pose.orientation = tf::createQuaternionMsgFromYaw(0);
        bbs_3d.markers.push_back(m);
    }
    bb_pub.publish(bbs_3d);
    last_n_bbs = n_bbs;

    // colored cloud
    typename pcl::PointCloud<t_p>::Ptr output_cloud(new pcl::PointCloud<t_p>);
    pcl_ros::transformPointCloud(*colored_cloud, *output_cloud, transform.inverse());
    sensor_msgs::PointCloud2 output_pc;
    pcl::toROSMsg(*output_cloud, output_pc);
    output_pc.header.frame_id = pc.header.frame_id;
    output_pc.header.stamp = ros::Time::now();
    pc_pub.publish(output_pc);

    // projection/depth image
    sensor_msgs::ImagePtr output_image;
    output_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", projection_image).toImageMsg();
    output_image->header.frame_id = image.header.frame_id;
    output_image->header.stamp = ros::Time::now();
    image_pub.publish(output_image);

    // edge detection
    cv::Mat edge_image;
    cv::cvtColor(bgr_image, edge_image, CV_RGB2GRAY);
    cv::GaussianBlur(edge_image, edge_image, cv::Size(5, 5), 0);
    cv::Canny(edge_image, edge_image, 100, 200, 3);
    /*
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edge_image, lines, 1, M_PI / 180, 20, 2, 10);
    for(int i=0;i<lines.size();i++){
        cv::Vec4i l = lines[i];
        cv::line(bgr_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3, CV_AA);
    }
    */
    cv::namedWindow("edge");
    cv::imshow("edge", edge_image);
    cv::waitKey(1);

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
void ObjectDetector3D<t_p>::get_euclidean_cluster(typename pcl::PointCloud<t_p>::Ptr& pc, pcl::PointCloud<t_p>& output_pc)
{
    //std::cout << "original points size: " << pc.points.size() << std::endl;
    if(pc->points.empty()){
        return;
    }

    // all input points have same color and intensity
    int r, g, b;
    r = pc->points.begin()->r;
    g = pc->points.begin()->g;
    b = pc->points.begin()->b;

    typename pcl::PointCloud<t_p>::Ptr cloud_filtered(new pcl::PointCloud<t_p>);
    pcl::copyPointCloud(*pc, *cloud_filtered);

    /*
    typename pcl::VoxelGrid<t_p> vg;
    vg.setInputCloud(cloud_filtered);
    vg.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
    vg.filter(*cloud_filtered);
    */

    // z to 0
    /*
    size_t n_points = cloud_filtered->points.size();
    std::vector<double> original_z(n_points);
    for(size_t i=0;i<n_points;++i){
        original_z[i] = cloud_filtered->points[i].z;
        cloud_filtered->points[i].z = 0;
    }
    */

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
    /*
    for(size_t i=0;i<n_points;++i){
        cloud_filtered->points[i].z = original_z[i];
    }
    */
    if(!cluster_indices.size() > 0){
        std::cout << "!!! clustering error !!!" << std::endl;
        //return;
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
        //std::cout << "cluster" << index << " size: " << cluster_size << std::endl;
        if(cluster_size > max_cluster_size){
            max_cluster_index = index;
            max_cluster_size = cluster_size;
            output_pc = *cluster;
        }
        index++;
    }
    std::cout << "final cluster size: " << output_pc.points.size() << std::endl;
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
void ObjectDetector3D<t_p>::coloring_pointcloud(typename pcl::PointCloud<t_p>::Ptr& cloud, int r, int g, int b)
{
    for(auto& pt : cloud->points){
        pt.r = r;
        pt.g = g;
        pt.b = b;
    }
}

template<typename t_p>
void ObjectDetector3D<t_p>::get_closest_point(typename pcl::PointCloud<t_p>::Ptr& cluster, t_p& closest_point)
{
    // camera optical frame
    double min_dist = 100;
    for(auto pt : cluster->points){
        double distance = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if(min_dist >= distance){
            min_dist = distance;
            closest_point = pt;
        }
    }
}

template<typename t_p>
void ObjectDetector3D<t_p>::principal_component_analysis(typename pcl::PointCloud<t_p>::Ptr& cluster, double& yaw)
{
    // camera optical frame
    double cluster_size = cluster->points.size();
    if(cluster_size < 3){
        return;
    }
    double ave_x = 0;
    double ave_y = 0;
    for(auto& pt : cluster->points){
        ave_x += pt.x;
        ave_y += pt.z;
    }
    ave_x /= cluster_size;
    ave_y /= cluster_size;
    double sigma_xx = 0;
    double sigma_xy = 0;
    double sigma_yy = 0;
    for(auto& pt : cluster->points){
        sigma_xx += (pt.x - ave_x) * (pt.x - ave_x);
        sigma_xy += (pt.x - ave_x) * (pt.z - ave_y);
        sigma_yy += (pt.z - ave_y) * (pt.z - ave_y);
    }
    sigma_xx /= cluster_size;
    sigma_xy /= cluster_size;
    sigma_yy /= cluster_size;
    Eigen::Matrix2d cov_mat;
    cov_mat << sigma_xx, sigma_xy,
               sigma_xy, sigma_yy;
    Eigen::EigenSolver<Eigen::Matrix2d> es(cov_mat);
    Eigen::Vector2d eigen_values = es.eigenvalues().real();
    Eigen::Matrix2d eigen_vectors = es.eigenvectors().real();
    std::cout << ave_x << ", " << ave_y << std::endl;
    std::cout << cov_mat << std::endl;
    std::cout << eigen_values << std::endl;
    std::cout << eigen_vectors << std::endl;
    int larger_index = 0;
    if(eigen_values(0) > eigen_values(1)){
        larger_index = 0;
    }else{
        larger_index = 1;
    }
    Eigen::Vector2d larger_vector = eigen_vectors.col(larger_index);
    yaw = atan2(larger_vector(1), larger_vector(0));
    std::cout << yaw << "[rad]" << std::endl;
}
