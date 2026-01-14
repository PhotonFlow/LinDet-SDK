#include <functional>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class DummyProcessorNode : public rclcpp::Node
{
public:
  DummyProcessorNode()
  : rclcpp::Node("ai_processor")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/left/image_raw",
      rclcpp::SensorDataQoS(),
      std::bind(&DummyProcessorNode::image_callback, this, std::placeholders::_1));
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    (void)msg;
    RCLCPP_INFO(this->get_logger(), "Processed Frame...");
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DummyProcessorNode>());
  rclcpp::shutdown();
  return 0;
}

