# Spear Head Tracker

这个库用于做杆头追踪。当前主方案是 heatmap 分割模型：模型从输入图片中预测杆头所在区域的 heatmap 和像素置信度，再把 heatmap 中超过阈值的连续区域转换为最小外接框，并用 `heatmap * confidence` 的区域平均值选择最可信的框。

## C++ 推理调用

C++ 推理代码在根目录的 `heatmap_cpp_infer.cpp` 中，依赖 LibTorch 和 OpenCV。

默认推荐的 TorchScript 模型路径是：

```text
runs/heatmap_train/heatmap_model.pt
```

如果还没有导出，可以先运行：

```bash
python model_code/export_heatmap_pt.py --weights runs/heatmap_train/best.pth --output runs/heatmap_train/heatmap_model.pt --img-size 384 384
```

C++ 侧提供两个函数：

```cpp
torch::jit::script::Module loadHeatmapModel(
    const std::string& model_path,
    torch::Device device = torch::kCPU);
```

用于从指定路径读取 heatmap `.pt` 模型。

```cpp
std::vector<std::array<float, 4>> predictHeatmapBoxes(
    torch::jit::script::Module& model,
    const std::string& image_path,
    int input_height = 384,
    int input_width = 384,
    float threshold = 0.5f,
    int min_area = 1,
    torch::Device device = torch::kCPU);
```

用于输入模型和图片路径，输出综合平均置信度最高的分割区域框。返回类型仍是 list；如果没有有效区域则为空，如果有结果则只包含一个框。每个框格式为：

```text
{center_x, center_y, width, height}
```

坐标是原图像素坐标，不是归一化坐标。

示例：

```cpp
#include <iostream>
#include "heatmap_cpp_infer.cpp"

int main() {
    auto device = torch::kCPU;
    auto model = loadHeatmapModel("runs/heatmap_train/heatmap_model.pt", device);

    auto boxes = predictHeatmapBoxes(
        model,
        "test.jpg",
        384,
        384,
        0.5f,
        10,
        device);

    for (const auto& box : boxes) {
        std::cout
            << "cx=" << box[0]
            << " cy=" << box[1]
            << " w=" << box[2]
            << " h=" << box[3]
            << std::endl;
    }
}
```

`threshold` 控制 heatmap 二值化阈值。`min_area` 用来过滤太小的噪声连通域。多个候选连通域存在时，C++ 推理会计算每个区域内 `heatmap_prob * confidence_prob` 的平均值，并只返回平均值最高的框。

## Heatmap Model 原理

Heatmap 模型定义在 `model_code/heatmap_model.py`。

模型结构：

1. 使用可选 backbone 提取图像特征。
2. backbone 可选：`resnet`、`darknet`、`mobilenet`、`efficientnet`。
3. heatmap head 使用卷积层把特征转换成双通道 logits。
4. 输出通过双线性插值恢复到输入图大小。

模型输出形状：

```text
[batch, 2, height, width]
```

第 0 个通道是 heatmap logits，第 1 个通道是 confidence logits。训练时两个通道都用 BCE 监督，目标都是预处理生成的 heatmap label。推理时对 logits 做 `sigmoid`，heatmap 通道负责决定哪些像素是前景，confidence 通道负责给前景区域打分。最终用 `heatmap_prob * confidence_prob` 的区域平均值选择最可信 bbox。

## Label 生成逻辑

Heatmap 数据集逻辑在 `model_code/heatmap_dataset.py`。

输入数据使用 YOLO 格式：

```text
class_id x_center y_center width height
```

其中 bbox 坐标是归一化坐标。

每张图片会被 resize 到训练尺寸，然后生成对应 heatmap label：

1. bbox 外部像素全部标记为 `0`。
2. bbox 内部会先根据颜色生成黑到灰色的 mask。
3. 通过 mask 的像素标记为 `1`。
4. bbox 内被 mask 筛掉的像素，会根据它到最近通过 mask 像素的距离获得一个高斯衰减值。
5. 超过 `max_fill_distance` 的像素会重新置为 `0`。
6. soft label 会再乘以 bbox 中心先验，减少 bbox 边缘和外部反光区域的影响。

生成后的图片和 heatmap 会缓存为 `.npy`，默认存到 `data/`。如果对应缓存已经存在，会直接读取缓存，避免每次训练重复生成。

## 训练参数

训练入口：

```bash
python model_code/heatmap_train.py --data dataset_6/data.yaml
```

常用参数：

`--data`：YOLO 数据集的 `data.yaml` 路径。

`--cache-dir`：`.npy` 缓存目录，默认 `data`。

`--output-dir`：训练输出目录，默认 `runs/heatmap_train`。会保存 `best.pth` 和 `latest.pth`。

`--resume`：指定 checkpoint 继续训练。

`--img-size H W`：训练和测试输入尺寸。

`--backbone`：选择 backbone，可选 `resnet`、`darknet`、`mobilenet`、`efficientnet`。

`--width-mult`：控制部分 backbone 的通道宽度。

`--head-channels`：heatmap head 的中间通道数。

`--epochs`：训练轮数。

`--batch-size`：batch 大小。

`--lr`：学习率。

`--weight-decay`：AdamW 权重衰减。

`--num-workers`：DataLoader worker 数。

`--gray-threshold`：黑到灰色像素筛选阈值。越小越严格，越大越容易包含亮灰色区域。

`--gaussian-sigma`：高斯填充的衰减速度。越大，反光断裂区域会被填得更宽。

`--max-fill-distance`：高斯填充的最大距离。超过该距离的像素置为 `0`，用于限制非目标反光区域扩散。

`--center-prior-sigma`：bbox 中心先验强度。越小越强调 bbox 中心，越能压低边缘区域。

`--pos-weight`：BCE 正样本权重。正样本很少时可以调大，避免模型全部预测背景。

`--conf-loss-weight`：confidence head 的 loss 权重。越大越强调像素置信度学习。

`--test-split`：每个 epoch 后可视化使用的 split，默认 `test`。

`--test-threshold`：测试时模型 heatmap 的二值化阈值。

`--test-label-threshold`：黄色 label heatmap 框的二值化阈值。不设置时使用 `--test-threshold`。

`--test-output`：bbox 可视化输出路径。

`--test-mask-output`：heatmap mask 可视化输出路径。

训练结束后，可以导出 C++ 可加载的 TorchScript：

```bash
python model_code/export_heatmap_pt.py --weights runs/heatmap_train/best.pth --output runs/heatmap_train/heatmap_model.pt --img-size 384 384
```
