#include "gemini_utils/uncompressor.hpp"

using namespace std::placeholders;

namespace gemini_utils
{

Uncompressor::Uncompressor(const rclcpp::NodeOptions & options)
: Node("uncompress_node", options)
{
  declare_parameter<std::string>("input_topic", "/camera/compressed");
  declare_parameter<std::string>("output_topic", "/camera/raw");
  declare_parameter<int>("queue_size", 10);

  get_parameter("input_topic", input_topic_);
  get_parameter("output_topic", output_topic_);
  get_parameter("queue_size", queue_size_);

  sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
    input_topic_, queue_size_,
    std::bind(&Uncompressor::topic_callback, this, _1)
  );

  pub_ = create_publisher<sensor_msgs::msg::Image>(
    output_topic_, rclcpp::QoS(queue_size_)
  );

  RCLCPP_INFO(get_logger(), "Subscribing to '%s', publishing to '%s'",
    input_topic_.c_str(), output_topic_.c_str());
}

void Uncompressor::topic_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
{
  try {
    // Convert buffer to cv::Mat
    std::vector<uint8_t> buf(msg->data.begin(), msg->data.end());
    cv::Mat image = cv::imdecode(buf, cv::IMREAD_COLOR);

    auto img_msg = cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
    pub_->publish(*img_msg);
  } catch (const cv::Exception & e) {
    RCLCPP_ERROR(get_logger(), "Failed to decode image: %s", e.what());
  }
}

}  // namespace gemini_utils

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<gemini_utils::Uncompressor>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}