#include <array>
#include <limits>
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

    torch::Tensor probabilities = torch::sigmoid(model.forward(inputs).toTensor())
                                      .squeeze(0)
                                      .detach()
                                      .to(torch::kCPU)
                                      .contiguous();
    if (probabilities.size(0) < 2) {
        throw std::runtime_error("Heatmap model must output 2 channels: heatmap and confidence.");
    }
    torch::Tensor heatmap = probabilities[0].contiguous();
    torch::Tensor confidence = probabilities[1].contiguous();
    torch::Tensor combined_confidence = (heatmap * confidence).contiguous();
    torch::Tensor mask_tensor = heatmap.gt(threshold).to(torch::kU8).contiguous();

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
    float best_score = -std::numeric_limits<float>::infinity();
    BoxXYWH best_box = {0.0f, 0.0f, 0.0f, 0.0f};
    const float* confidence_data = combined_confidence.data_ptr<float>();

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

        float score_sum = 0.0f;
        int score_count = 0;
        for (int y = top; y < top + height; ++y) {
            const int* label_row = labels.ptr<int>(y);
            for (int x = left; x < left + width; ++x) {
                if (label_row[x] != label) {
                    continue;
                }
                score_sum += confidence_data[y * input_width + x];
                ++score_count;
            }
        }
        if (score_count == 0) {
            continue;
        }

        const float mean_score = score_sum / static_cast<float>(score_count);
        if (mean_score <= best_score) {
            continue;
        }

        best_score = mean_score;
        best_box = {
            (x1 + x2) * 0.5f,
            (y1 + y2) * 0.5f,
            x2 - x1,
            y2 - y1,
        };
    }

    if (best_score > -std::numeric_limits<float>::infinity()) {
        boxes.push_back(best_box);
    }

    return boxes;
}
