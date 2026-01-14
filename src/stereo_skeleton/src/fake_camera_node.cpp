#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

class FakeCameraNode : public rclcpp::Node
{
public:
  FakeCameraNode()
  : rclcpp::Node("fake_camera_driver")
  {
    left_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/left/image_raw", rclcpp::SensorDataQoS());
    right_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/right/image_raw", rclcpp::SensorDataQoS());

    init_image(left_image_, "fake_left_camera");
    init_image(right_image_, "fake_right_camera");

    timer_ = this->create_wall_timer(
      33ms, std::bind(&FakeCameraNode::publish_images, this));
  }

private:
  void init_image(sensor_msgs::msg::Image & image, const std::string & frame_id)
  {
    const uint32_t width = 1280;
    const uint32_t height = 720;

    image.header.frame_id = frame_id;
    image.height = height;
    image.width = width;
    image.encoding = "mono8";
    image.is_bigendian = false;
    image.step = width;
    image.data.assign(width * height, 0);
  }

  void publish_images()
  {
    const auto stamp = this->now();
    left_image_.header.stamp = stamp;
    right_image_.header.stamp = stamp;

    left_pub_->publish(left_image_);
    right_pub_->publish(right_image_);
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  sensor_msgs::msg::Image left_image_;
  sensor_msgs::msg::Image right_image_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FakeCameraNode>());
  rclcpp::shutdown();
  return 0;
}

