#ifndef GEMINI_UTILS_UNCOMPRESSOR_HPP_
#define GEMINI_UTILS_UNCOMPRESSOR_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>      // ← header corregido
#include <opencv2/opencv.hpp>

namespace gemini_utils
{
class Uncompressor : public rclcpp::Node
{
public:
  explicit Uncompressor(const rclcpp::NodeOptions & options);

private:
  void topic_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;

  std::string input_topic_;
  std::string output_topic_;
  int queue_size_;
};
}  // namespace gemini_utils

#endif  // GEMINI_UTILS_UNCOMPRESSOR_HPP_
