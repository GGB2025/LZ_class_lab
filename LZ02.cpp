// 作业：统计 pin.png 中回形针的数量
// 处理流程：灰度化 -> 高斯平滑 -> Otsu 自适应阈值二值化 -> 形态学开运算
//           -> 提取外轮廓 -> 按面积筛选 -> 计数并在原图上标注
// 回形针在图中是深色细线，背景接近白色，因此用“反向阈值”把回形针变成白色前景。

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    // 默认读取 pin.png，也可以在命令行传入其它图片路径
    const std::string imagePath = (argc > 1) ? argv[1] : "pin.png";

    try
    {
        cv::Mat src = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (src.empty())
        {
            std::cerr << "无法读取图片: " << imagePath << std::endl;
            std::cerr << "请检查图片路径和当前工作目录。" << std::endl;
            return 1;
        }

        // 1. 转灰度图：计数只需要明暗信息，单通道处理更快
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

        // 2. 高斯平滑：抑制图像噪声，使后面得到的二值轮廓更干净
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        // 3. Otsu 自动阈值 + 反向二值化：
        //    背景亮、回形针暗，反向之后回形针是白色（255），背景是黑色（0）
        cv::Mat binary;
        cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        // 4. 形态学开运算：先腐蚀后膨胀，去掉二值图中的小噪点，
        //    同时保持回形针本身的形状和面积基本不变
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

        // 5. 提取外轮廓：RETR_EXTERNAL 只取最外层轮廓，
        //    回形针内部的小孔不会被重复统计
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 6. 按面积筛选轮廓：
        //    - 图像边缘的黑框等干扰面积较小，会被下限过滤掉；
        //    - 单个回形针外轮廓面积约 10000 像素，落在区间内；
        //    - 两个回形针粘连时面积会接近两倍，这里也用它来检查异常。
        const double minArea = 5000.0;    // 轮廓面积下限
        const double maxArea = 50000.0;   // 轮廓面积上限

        cv::Mat result = src.clone();
        int clipCount = 0;

        for (size_t i = 0; i < contours.size(); ++i)
        {
            const double area = cv::contourArea(contours[i]);
            if (area < minArea || area > maxArea)
                continue;

            const cv::Rect box = cv::boundingRect(contours[i]);
            if (box.width < 40 || box.height < 40)
                continue;

            ++clipCount;

            // 用绿色画出回形针轮廓，用红色画出外接矩形
            cv::drawContours(result, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
            cv::rectangle(result, box, cv::Scalar(0, 0, 255), 2);

            // 在矩形左上角标出序号
            const std::string label = std::to_string(clipCount);
            cv::putText(result, label, box.tl() + cv::Point(2, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        }

        // 7. 在结果图左上角显示总数
        const std::string summary = "paperclips: " + std::to_string(clipCount);
        cv::putText(result, summary, cv::Point(12, 32),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);

        std::cout << "图片: " << imagePath << " (" << src.cols << " x " << src.rows << ")" << std::endl;
        std::cout << "回形针数量: " << clipCount << std::endl;

        // 8. 显示原图、二值掩膜和计数结果，按任意键退出
        cv::namedWindow("LZ02 - Original", cv::WINDOW_NORMAL);
        cv::namedWindow("LZ02 - Binary", cv::WINDOW_NORMAL);
        cv::namedWindow("LZ02 - Result", cv::WINDOW_NORMAL);
        cv::imshow("LZ02 - Original", src);
        cv::imshow("LZ02 - Binary", binary);
        cv::imshow("LZ02 - Result", result);
        std::cout << "在任意图像窗口中按键盘键退出。" << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const cv::Exception& error)
    {
        std::cerr << "OpenCV 错误: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}