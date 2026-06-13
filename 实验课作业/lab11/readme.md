# 实验室11
## ✨ 项目文件列表
- lab11task1.py          # 任务1：视频转骨架npy数据集生成
- lab11task2.py          # 任务2：Transformer模型训练（输出曲线、混淆矩阵、最优权重）
- lab11task3.py          # 任务3：单视频推理预测
- lab11add.py            # 附加：Transformer多头注意力热力图分析
- vis_skeleton_final.py  # 骨架视频可视化渲染脚本
- X_train.npy            # 训练集骨架时序数据
- X_test.npy             # 测试集骨架时序数据
- y_train.npy            # 训练集动作标签
- y_test.npy             # 测试集动作标签
- label_map.json         # 6类动作名称与数字标签映射字典
- badminton_transformer.pth       # 推理专用最终模型权重
- badminton_transformer_best.pth  # 训练全程最高精度备份权重
- best_model.pth                 # 早停机制保存的中间轮次最优权重
- confusion_matrix.png    # 6分类混淆矩阵热力图
- skeleton_lift.mp4       # 正手挑球（识别效果最好类别）骨架标注对比视频
- skeleton_net.mp4        # 正手网前小球（识别效果最差类别）骨架标注对比视频
- lab11实验报告.docx      # 完整Word实验报告文档
- README.md               # 项目说明文档（当前正在编辑的文件）
