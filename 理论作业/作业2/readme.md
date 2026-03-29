
# 作业2
基于 Python + OpenCV 实现图像增强课程实验，对低对比度、含噪声、细节丰富三类图像进行处理。

## 实现功能
- 手动实现全局直方图均衡化
- CLAHE 自适应对比度增强
- 均值滤波、高斯滤波、中值滤波
- 拉普拉斯锐化
- 组合处理：滤波→均衡、均衡→滤波
- 自动生成对比图与直方图
- 定量评价：PSNR、信息熵

## 环境配置
pip install opencv-python numpy matplotlib

## 使用方法
1. 准备 3 张图片并命名：
   - img1.jpg：低对比度图像
   - img2.jpg：含噪声图像
   - img3.jpg：细节纹理丰富图像
2. 将图片与代码放在同一目录
3. 运行代码，自动生成结果图

## 文件结构
- image_enhancement.py  主程序
- img1.jpg、img2.jpg、img3.jpg  输入图像
- 输出：各图片对比图与直方图
