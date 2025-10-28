# 创建目标目录结构
mkdir -p /root/autodl-tmp/AetherNet/datasets/train/GT
mkdir -p /root/autodl-tmp/AetherNet/datasets/train/LQ
mkdir -p /root/autodl-tmp/AetherNet/datasets/valid/GT
mkdir -p /root/autodl-tmp/AetherNet/datasets/valid/LQ

#!/bin/bash
DATASET_DIR="/root/autodl-tmp/AetherNet/datasets"
SOURCE_DIR="/root/autodl-pub/DIV2K"

echo "创建目录结构..."
mkdir -p $DATASET_DIR/train/GT
mkdir -p $DATASET_DIR/train/LQ
mkdir -p $DATASET_DIR/valid/GT
mkdir -p $DATASET_DIR/valid/LQ

# 创建临时解压目录
TEMP_DIR="/root/autodl-tmp/temp_div2k"
mkdir -p $TEMP_DIR

echo "步骤1: 解压高分辨率图像..."
# 解压训练集HR
echo "解压训练集高分辨率图像..."
unzip -q $SOURCE_DIR/HighResolution/DIV2K_train_HR.zip -d $TEMP_DIR/train_hr
# 解压验证集HR
echo "解压验证集高分辨率图像..."
unzip -q $SOURCE_DIR/HighResolution/DIV2K_valid_HR.zip -d $TEMP_DIR/valid_hr

echo "步骤2: 解压低分辨率图像（选择bicubic X4倍下采样）..."
# 解压训练集LR
echo "解压训练集低分辨率图像..."
unzip -q $SOURCE_DIR/LowRes2017/DIV2K_train_LR_bicubic_X4.zip -d $TEMP_DIR/train_lr
# 解压验证集LR
echo "解压验证集低分辨率图像..."
unzip -q $SOURCE_DIR/LowRes2017/DIV2K_valid_LR_bicubic_X4.zip -d $TEMP_DIR/valid_lr

echo "步骤3: 整理文件到目标结构..."
# 移动高分辨率图像
echo "移动高分辨率训练图像..."
mv $TEMP_DIR/train_hr/DIV2K_train_HR/* $DATASET_DIR/train/GT/ 2>/dev/null || mv $TEMP_DIR/train_hr/* $DATASET_DIR/train/GT/
echo "移动高分辨率验证图像..."
mv $TEMP_DIR/valid_hr/DIV2K_valid_HR/* $DATASET_DIR/valid/GT/ 2>/dev/null || mv $TEMP_DIR/valid_hr/* $DATASET_DIR/valid/GT/

# 移动低分辨率图像
echo "移动低分辨率训练图像..."
mv $TEMP_DIR/train_lr/DIV2K_train_LR_bicubic/X4/* $DATASET_DIR/train/LQ/ 2>/dev/null || mv $TEMP_DIR/train_lr/* $DATASET_DIR/train/LQ/
echo "移动低分辨率验证图像..."
mv $TEMP_DIR/valid_lr/DIV2K_valid_LR_bicubic/X4/* $DATASET_DIR/valid/LQ/ 2>/dev/null || mv $TEMP_DIR/valid_lr/* $DATASET_DIR/valid/LQ/

echo "步骤4: 清理临时文件..."
rm -rf $TEMP_DIR

echo "步骤5: 验证文件数量..."
echo "训练集 GT: $(ls $DATASET_DIR/train/GT | wc -l) 张图像"
echo "训练集 LQ: $(ls $DATASET_DIR/train/LQ | wc -l) 张图像"
echo "验证集 GT: $(ls $DATASET_DIR/valid/GT | wc -l) 张图像"
echo "验证集 LQ: $(ls $DATASET_DIR/valid/LQ | wc -l) 张图像"

echo "数据集整理完成！"
echo "最终结构："
echo "$DATASET_DIR/"
echo "├── train/"
echo "│   ├── GT/    # 高分辨率训练图像"
echo "│   └── LQ/    # 低分辨率训练图像"
echo "└── valid/"
echo "    ├── GT/    # 高分辨率验证图像"
echo "    └── LQ/    # 低分辨率验证图像"