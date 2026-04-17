import cv2
import numpy as np
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


# ====================== 1. 测试图像 & 三种变换 ======================
def create_test_image():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), 2)
    cv2.circle(img, (300, 300), 50, (0, 255, 0), 2)
    cv2.line(img, (50, 250), (350, 250), (255, 0, 0), 2)
    cv2.line(img, (50, 280), (350, 280), (255, 0, 0), 2)
    cv2.line(img, (100, 50), (100, 350), (0, 0, 0), 2)
    cv2.line(img, (150, 50), (150, 350), (0, 0, 0), 2)
    return img

def similarity_transform(img):
    rows, cols = img.shape[:2]
    center = (cols//2, rows//2)
    M = cv2.getRotationMatrix2D(center, 30, 0.8)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def affine_transform(img):
    rows, cols = img.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[80, 100], [220, 80], [60, 230]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def perspective_transform(img):
    rows, cols = img.shape[:2]
    pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    pts2 = np.float32([[50, 50], [cols-50, 80], [30, rows-30], [cols-30, rows-50]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst
# ====================== 全自动文档校正（超强版） ======================
def auto_correct_paper(image_path):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图片")
    
    # 2. 预处理：灰度 + 模糊 + 边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 3. 找轮廓
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # 4. 寻找四边形（纸张四个角）
    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is None:
        raise Exception("未检测到纸张边缘，请确保背景干净、光线充足")

    # 5. 四点排序（必须正确：左上 → 右上 → 右下 → 左下）
    pts = screen_cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    # 6. 目标尺寸（标准A4）
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 目标坐标
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 7. 透视变换
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return img, warped


if __name__ == "__main__":
    # 1. 生成三种变换结果
    print("正在生成三种变换对比图...")
    test_img = create_test_image()
    sim = similarity_transform(test_img)
    aff = affine_transform(test_img)
    per = perspective_transform(test_img)

    # 拼接并保存，不用plt
    top = np.hstack((test_img, sim))
    bottom = np.hstack((aff, per))
    all_transform = np.vstack((top, bottom))
    cv2.imwrite("transform_result.png", all_transform)
    print("三种变换对比图已生成！")

    
if __name__ == "__main__":
    img_path = "paper.jpg"

    if not os.path.exists(img_path):
        print(f"请把图片命名为 paper.jpg 放在同一目录")
    else:
       
        original, corrected = auto_correct_paper(img_path)

        # 保存结果
        cv2.imwrite("corrected_paper.jpg", corrected)
        
        # 生成对比图
        h = original.shape[0]
        corrected_resized = cv2.resize(corrected, (int(corrected.shape[1] * h / corrected.shape[0]), h))
        compare = np.hstack((original, corrected_resized))
        cv2.imwrite("paper_corrected.png", compare)

        print("校正完成！")
        print("corrected_paper.jpg = 校正后的平整A4")
        print("paper_corrected.png = 校正前后对比")

        '''
三种几何变换对几何性质的影响
1. 相似变换
直线：保持为直线。
平行线：仍然保持平行。
垂直线：变换后依然保持垂直关系。
圆：仍然保持为正圆，仅位置、角度或整体大小发生变化，形状不改变。
2. 仿射变换
直线：保持为直线。
平行线：变换后仍然保持平行关系。
垂直线：变换后不再保持垂直，夹角会发生改变。
圆：会被拉伸或压缩，变成椭圆，不再是标准圆形。
3. 透视变换
直线：保持为直线。
平行线：变换后不再平行，会向消失点汇聚。
垂直线：变换后不再保持垂直，夹角会发生改变。
        '''