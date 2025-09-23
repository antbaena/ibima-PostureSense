#include "gemini_utils/depth_decompressor.hpp"



using namespace std::placeholders;

namespace gemini_utils
{

DepthDecompressor::DepthDecompressor(const rclcpp::NodeOptions & options)
: Node("depth_decompressor_node", options)
{
  declare_parameter<std::string>("depth_input_topic", "/cam00/camera_00/depth/image_raw/compressedDepth");
  declare_parameter<std::string>("depth_output_topic", "/cam00/camera_00/depth/image_raw");
  declare_parameter<int>("queue_size", 10);

  get_parameter("depth_input_topic", input_topic_);
  get_parameter("depth_output_topic", output_topic_);
  get_parameter("queue_size", queue_size_);

  sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
    input_topic_, queue_size_,
    std::bind(&DepthDecompressor::topic_callback, this, _1)
  );

  pub_ = create_publisher<sensor_msgs::msg::Image>(
    output_topic_, rclcpp::QoS(queue_size_)
  );

  RCLCPP_INFO(get_logger(), "Subscribing to '%s', publishing to '%s'",
    input_topic_.c_str(), output_topic_.c_str());
}

void DepthDecompressor::topic_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
{
  try {
    std::string format_str = msg->format;
    std::string depth_fmt;

    // Parse: split by ';'
    auto semicolon_pos = format_str.find(';');
    if (semicolon_pos == std::string::npos) {
      throw std::runtime_error("Unexpected format string: missing ';' in '" + format_str + "'");
    }

    depth_fmt = format_str.substr(0, semicolon_pos);
    std::string compression_info = format_str.substr(semicolon_pos + 1);

    // Trim spaces
    depth_fmt.erase(depth_fmt.find_last_not_of(" \n\r\t")+1);
    depth_fmt.erase(0, depth_fmt.find_first_not_of(" \n\r\t"));
    compression_info.erase(0, compression_info.find_first_not_of(" \n\r\t"));

    // Check if "compressedDepth" is in the string
    if (compression_info.find("compressedDepth") == std::string::npos) {
      throw std::runtime_error("Compression type is not 'compressedDepth'. Found: '" + compression_info + "'");
    }

    const size_t depth_header_size = 12;
    if (msg->data.size() <= depth_header_size) {
      throw std::runtime_error("Compressed depth message too small.");
    }

    // Remove header
    std::vector<uint8_t> raw_data(msg->data.begin() + depth_header_size, msg->data.end());
    cv::Mat depth_img_raw = cv::imdecode(raw_data, cv::IMREAD_UNCHANGED);

    if (depth_img_raw.empty()) {
      throw std::runtime_error("Could not decode compressed depth image. Header size might be incorrect.");
    }

    sensor_msgs::msg::Image::SharedPtr output_msg;

    if (depth_fmt == "16UC1") {
      output_msg = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::TYPE_16UC1, depth_img_raw).toImageMsg();
    } else if (depth_fmt == "32FC1") {
      const uint8_t * raw_header = msg->data.data();
      int compfmt;
      float depthQuantA, depthQuantB;
      std::memcpy(&compfmt, raw_header, sizeof(int));
      std::memcpy(&depthQuantA, raw_header + 4, sizeof(float));
      std::memcpy(&depthQuantB, raw_header + 8, sizeof(float));

      cv::Mat depth_float;
      depth_img_raw.convertTo(depth_float, CV_32FC1);

      cv::Mat depth_scaled = depthQuantA / (depth_float - depthQuantB);
      depth_scaled.setTo(0, depth_img_raw == 0);

      cv::Mat depth_mm;
      depth_scaled.convertTo(depth_mm, CV_16UC1, 1000.0);  // convert to mm

      output_msg = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::TYPE_16UC1, depth_mm).toImageMsg();
    } else {
      throw std::runtime_error("Unsupported depth format: " + depth_fmt);
    }

    pub_->publish(*output_msg);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Failed to decode depth image: %s", e.what());
  }
}

}  // namespace gemini_utils

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<gemini_utils::DepthDecompressor>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
