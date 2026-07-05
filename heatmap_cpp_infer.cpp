#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>

using BoxXYWH = std::array<float, 4>;

torch::jit::script::Module loadHeatmapModel(
    const std::string& model_path,
    torch::Device device = torch::kCPU) {
    torch::jit::script::Module model = torch::jit::load(model_path, device);
    model.eval();
    return model;
}

std::vector<BoxXYWH> predictHeatmapBoxes(
    torch::jit::script::Module& model,
    const std::string& image_path,
    int input_height = 384,
    int input_width = 384,
    float threshold = 0.5f,
    int min_area = 1,
    torch::Device device = torch::kCPU) {
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to read image: " + image_path);
    }

    const int original_height = bgr.rows;
    const int original_width = bgr.cols;

    cv::Mat resized_bgr;
    cv::resize(bgr, resized_bgr, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized_bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat rgb_float;
    rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

    torch::Tensor input = torch::from_blob(
                              rgb_float.data,
                              {1, input_height, input_width, 3},
                              torch::TensorOptions().dtype(torch::kFloat32))
                              .permute({0, 3, 1, 2})
                              .contiguous()
                              .to(device);

    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    torch::Tensor logits = model.forward(inputs).toTensor();
    torch::Tensor mask_tensor = torch::sigmoid(logits)
                                    .squeeze()
                                    .detach()
                                    .to(torch::kCPU)
                                    .gt(threshold)
                                    .to(torch::kU8)
                                    .contiguous();

    cv::Mat mask(input_height, input_width, CV_8UC1, mask_tensor.data_ptr<unsigned char>());
    cv::Mat mask_copy = mask.clone();

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        mask_copy,
        labels,
        stats,
        centroids,
        4,
        CV_32S);

    const float scale_x = static_cast<float>(original_width) / static_cast<float>(input_width);
    const float scale_y = static_cast<float>(original_height) / static_cast<float>(input_height);

    std::vector<BoxXYWH> boxes;
    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < min_area) {
            continue;
        }

        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        const float x1 = static_cast<float>(left) * scale_x;
        const float y1 = static_cast<float>(top) * scale_y;
        const float x2 = static_cast<float>(left + width) * scale_x;
        const float y2 = static_cast<float>(top + height) * scale_y;

        boxes.push_back({
            (x1 + x2) * 0.5f,
            (y1 + y2) * 0.5f,
            x2 - x1,
            y2 - y1,
        });
    }

    return boxes;
}
