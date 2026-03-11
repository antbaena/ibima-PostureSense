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
// QoS para cámaras = Best Effort + baja latencia
  auto qos = rclcpp::SensorDataQoS()
                .keep_last(queue_size_)
                .reliability(rclcpp::ReliabilityPolicy::BestEffort);

  sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
    input_topic_, qos,
    std::bind(&Uncompressor::topic_callback, this, _1)
  );

  pub_ = create_publisher<sensor_msgs::msg::Image>(
    output_topic_, qos
  );

  RCLCPP_INFO(get_logger(), "Subscribing to '%s', publishing to '%s'",
    input_topic_.c_str(), output_topic_.c_str());
}

void Uncompressor::topic_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
{
  try {
    if (msg->data.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Received empty compressed image on '%s'", input_topic_.c_str());
      return;
    }

    // Convert buffer to cv::Mat
    std::vector<uint8_t> buf(msg->data.begin(), msg->data.end());
    cv::Mat image = cv::imdecode(buf, cv::IMREAD_COLOR);

    if (image.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "imdecode returned empty Mat — corrupt or unsupported JPEG frame on '%s'",
        input_topic_.c_str());
      return;
    }

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