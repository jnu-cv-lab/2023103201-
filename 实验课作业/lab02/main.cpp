#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // -------------------------- 任务1：读取测试图片 --------------------------
    Mat src = imread("test.jpg"); // 替换为你的图片路径
    if (src.empty())
    {
        cout << "无法读取图片，请检查路径！" << endl;
        return -1;
    }

    // -------------------------- 任务2：输出图像基本信息 --------------------------
    cout << "=== 图像基本信息 ===" << endl;
    cout << "宽度 (Width): " << src.cols << " px" << endl;
    cout << "高度 (Height): " << src.rows << " px" << endl;
    cout << "通道数 (Channels): " << src.channels() << endl;
    cout << "数据类型 (Type): " << typeToString(src.type()) << endl;

    // -------------------------- 任务3：显示原图 --------------------------
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", src);

    // -------------------------- 任务4：转换为灰度图并显示 --------------------------
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    namedWindow("灰度图", WINDOW_AUTOSIZE);
    imshow("灰度图", gray);

    // -------------------------- 任务5：保存灰度图 --------------------------
    imwrite("c++ gray_test.jpg", gray);
    cout << "灰度图已保存为 c++ gray_test.jpg" << endl;

    // -------------------------- 任务6：NumPy 风格简单操作（C++ 实现） --------------------------
    // 1. 输出某个像素值（以原图左上角 (0,0) 为例）
    Vec3b pixel = src.at<Vec3b>(0, 0);
    cout << "左上角像素 (BGR): ("
         << (int)pixel[0] << ", "
         << (int)pixel[1] << ", "
         << (int)pixel[2] << ")" << endl;

    // 2. 裁剪左上角区域并保存
    Rect roi(0, 0, 200, 200); // 从 (0,0) 开始，宽200，高200
    Mat cropped = src(roi);
    imwrite("c++ cropped_top_left.jpg", cropped);
    cout << "左上角裁剪图已保存为 c++ cropped_top_left.jpg" << endl;

    // 显示裁剪图
    namedWindow("裁剪图", WINDOW_AUTOSIZE);
    imshow("裁剪图", cropped);

    // 等待按键后关闭窗口
    waitKey(0);
    destroyAllWindows();

    return 0;
}