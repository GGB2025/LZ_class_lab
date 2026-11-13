// 作业：统计 penny.png 中硬币的数量
// 处理流程：灰度化 -> 高斯平滑 -> Otsu 自适应阈值二值化 -> 形态学闭/开运算
//           -> 提取外轮廓 -> 按面积筛选 -> 计数并在原图上标注
// 硬币在图中比深灰色背景亮得多，因此用“正向阈值”把硬币变成白色前景。

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    // 默认读取 penny.png，也可以在命令行传入其它图片路径
    const std::string imagePath = (argc > 1) ? argv[1] : "penny.png";

    try
    {
        cv::Mat src = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (src.empty())
        {
            std::cerr << "无法读取图片: " << imagePath << std::endl;
            std::cerr << "请检查图片路径和当前工作目录。" << std::endl;
            return 1;
        }

        // 1. 转灰度图：硬币是圆形金属片，明暗差异已经足够区分背景
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

        // 2. 高斯平滑：抑制硬币表面纹理和背景噪声
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

        // 3. Otsu 自动阈值 + 正向二值化：
        //    背景暗、硬币亮，阈值化之后硬币是白色（255），背景是黑色（0）
        cv::Mat binary;
        cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // 4. 形态学处理：
        //    闭运算填补硬币表面因反光/暗部造成的小空洞，
        //    开运算去掉背景中的零星亮点，硬币本身面积基本不变
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

        // 5. 提取外轮廓：RETR_EXTERNAL 只取最外层轮廓，
        //    硬币上的花纹、文字等内部结构不会被重复统计
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 6. 按面积筛选轮廓：
        //    - 图像四周的黑框和细小噪点面积很小，会被下限过滤掉；
        //    - 一枚硬币的外轮廓面积约 10000 像素（被遮挡的那枚只有约 4900），
        //      所以下限取 3000；上限用于排除多枚硬币粘连成大块的情况。
        const double minArea = 3000.0;    // 轮廓面积下限
        const double maxArea = 40000.0;   // 轮廓面积上限

        cv::Mat result = src.clone();
        int coinCount = 0;

        for (size_t i = 0; i < contours.size(); ++i)
        {
            const double area = cv::contourArea(contours[i]);
            if (area < minArea || area > maxArea)
                continue;

            const cv::Rect box = cv::boundingRect(contours[i]);
            if (box.width < 40 || box.height < 40)
                continue;

            ++coinCount;

            // 用绿色画出硬币轮廓
            cv::drawContours(result, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);

            // 用最小外接圆近似硬币的圆形边界，红色显示，圆心画成小圆点
            cv::Point2f center;
            float radius = 0.0f;
            cv::minEnclosingCircle(contours[i], center, radius);
            cv::circle(result, center, cvRound(radius), cv::Scalar(0, 0, 255), 2);
            cv::circle(result, center, 3, cv::Scalar(0, 0, 255), -1);

            // 在硬币上方标出序号
            const std::string label = std::to_string(coinCount);
            cv::putText(result, label, cv::Point(cvRound(center.x) - 8, cvRound(center.y - radius) - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
        }

        // 7. 在结果图左上角显示总数
        const std::string summary = "coins: " + std::to_string(coinCount);
        cv::putText(result, summary, cv::Point(12, 32),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);

        std::cout << "图片: " << imagePath << " (" << src.cols << " x " << src.rows << ")" << std::endl;
        std::cout << "硬币数量: " << coinCount << std::endl;

        // 8. 显示原图、二值掩膜和计数结果，按任意键退出
        cv::namedWindow("LZ02_2 - Original", cv::WINDOW_NORMAL);
        cv::namedWindow("LZ02_2 - Binary", cv::WINDOW_NORMAL);
        cv::namedWindow("LZ02_2 - Result", cv::WINDOW_NORMAL);
        cv::imshow("LZ02_2 - Original", src);
        cv::imshow("LZ02_2 - Binary", binary);
        cv::imshow("LZ02_2 - Result", result);
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