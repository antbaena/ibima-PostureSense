#ifndef GEMINI_UTILS__DEPTH_DECOMPRESSOR_HPP_
#define GEMINI_UTILS__DEPTH_DECOMPRESSOR_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <string>

namespace gemini_utils
{

class DepthDecompressor : public rclcpp::Node
{
public:
  explicit DepthDecompressor(const rclcpp::NodeOptions & options);

private:
  void topic_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg);

  std::string input_topic_;
  std::string output_topic_;
  int queue_size_;

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
};

}  // namespace gemini_utils

#endif  // GEMINI_UTILS__DEPTH_DECOMPRESSOR_HPP_
