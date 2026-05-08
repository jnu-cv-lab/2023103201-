# 导入所需库
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# 任务1：数据准备（作业要求）
# --------------------------

# 1. 加载sklearn自带的手写数字数据集
digits = load_digits()

# 2. 查看数据集中图像的数量
print("=" * 50)
print("数据集基本信息：")
print("图像总数：", len(digits.images))
print("标签总数：", len(digits.target))

# 3. 查看每张图像的大小
print("单张图像大小：", digits.images[0].shape)  # 输出 (8, 8)
print("展平后的特征向量维度：", digits.data[0].shape)  # 输出 (64,)

# 4. 查看类别标签
print("类别标签集合：", np.unique(digits.target))  # 输出 0~9
print("每类样本数量：", np.bincount(digits.target))

# 5. 显示若干张样本图像及其真实标签
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")
    plt.axis('off')
plt.tight_layout()
# 保存为当前目录下的 sample_digits.png
plt.savefig("01sample_digits.png", dpi=150)
print("✅ 样本图像已保存为 01sample_digits.png")


# 数据集划分（你需要的train_test_split部分）
# --------------------------
X = digits.data  # 特征矩阵 (1797, 64)
y = digits.target  # 标签向量 (1797,)

# 修正了变量名和函数名的空格问题
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\n训练集样本数：", len(X_train))
print("测试集样本数：", len(X_test))
print("=" * 50)


# --------------------------
# 任务2：数据划分（测试集比例25%）
# --------------------------
from sklearn.model_selection import train_test_split

# 加载数据（如果前面已经加载过，这行可以省略）
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target

# 数据划分：测试集占25%，随机种子固定保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 输出划分结果
print("=" * 50)
print("任务2：数据划分结果")
print(f"总样本数：{len(X)}")
print(f"训练集样本数：{len(X_train)} ({len(X_train)/len(X):.1%})")
print(f"测试集样本数：{len(X_test)} ({len(X_test)/len(X):.1%})")
print("=" * 50)


# --------------------------
# 任务3：特征表示
# --------------------------
from sklearn.datasets import load_digits
import numpy as np

# 加载数据
digits = load_digits()

# 取一张图像示例
img = digits.images[0]  # shape: (8, 8)
print("=" * 50)
print("任务3：特征表示转换示例")
print("原始图像形状：", img.shape)

# 1. 图像转特征向量（展平操作）
feature_vector = img.flatten()  # 或用 img.reshape(-1)
print("展平后的特征向量形状：", feature_vector.shape)
print("特征向量前10个值：", feature_vector[:10])

# 2. 说明：digits.data 已经是所有图像的展平结果
print("\ndigits.data 矩阵形状：", digits.data.shape)  # (1797, 64)
print("=" * 50)


# --------------------------
# 任务4：模型训练（含分类结果对比）
# --------------------------
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 导入所有模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 加载并划分数据（75%训练集，25%测试集）
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. 定义模型列表
models = [
    ("K近邻 (KNN)", KNeighborsClassifier()),
    ("朴素贝叶斯", GaussianNB()),
    ("逻辑回归", LogisticRegression(max_iter=1000)),
    ("支持向量机 (SVM)", SVC()),
    ("决策树", DecisionTreeClassifier(random_state=42)),
    ("随机森林", RandomForestClassifier(random_state=42))
]

print("=" * 60)
print("任务4：模型训练、分类预测与准确率")
print("=" * 60)

# 3. 逐个训练、预测并输出结果
for name, model in models:
    # 训练模型
    model.fit(X_train, y_train)
    # 分类预测
    y_pred = model.predict(X_test)
    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n【模型：{name}】")
    print(f"测试集准确率：{acc:.4f}")
    
    # 打印前10个测试样本的分类结果对比
    print("前10个测试样本分类结果（真实 vs 预测）：")
    print("-" * 40)
    for i in range(10):
        print(f"样本{i:2d} | 真实: {y_test[i]} | 预测: {y_pred[i]}")
    print("-" * 40)


    # --------------------------
# 任务5：结果比较
# --------------------------
print("\n任务5：不同模型测试准确率对比")
print("=" * 35)
print(f"{'模型':<20} | {'测试准确率':>10}")
print("-" * 35)
# 假设你之前已经把每个模型的准确率存到了一个字典里，比如：
# results = {"KNN": 0.9867, "Naive Bayes": 0.8400, ...}
# 或者直接在循环里记录：
results = {}
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

for name, acc in results.items():
    print(f"{name:<20} | {acc:>10.4f}")
print("=" * 35)


# --------------------------
# 任务6：错误样本分析（以SVM为例）
# --------------------------
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载并划分数据
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. 训练表现最好的SVM模型
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# 3. 绘制混淆矩阵
print("=" * 50)
print("任务6：错误样本分析（SVM模型）")
print("=" * 50)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵：")
print(cm)

# 绘制混淆矩阵（保存为图片）
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (SVM)")
plt.savefig("06confusion_matrix.png", dpi=150)
print("\n✅ 混淆矩阵已保存为 06confusion_matrix.png")

# 4. 找出所有被错误分类的样本
wrong_indices = np.where(y_pred != y_test)[0]
print(f"\n错误分类的样本总数：{len(wrong_indices)}")

# 显示前5个错误分类的样本
if len(wrong_indices) > 0:
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(wrong_indices[:5]):
        plt.subplot(1, 5, i + 1)
        # 还原8×8图像
        img = X_test[idx].reshape(8, 8)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("06wrong_samples.png", dpi=150)
    print("✅ 错误样本图像已保存为 06wrong_samples.png")
else:
    print("无错误分类样本")