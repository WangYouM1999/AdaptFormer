#!/bin/bash

folder_path1="./model_weights/massbuilding/buildformer"  # 替换为实际的权重路径
folder_path2="./fig_results"  # 替换为实际的diff文件路径

echo "-------开始删除权重-------"
find $folder_path1 -type f \( -name "*.ckpt" -o -name "buildformer*" -o -name "last*" \) -print -delete
echo "-------删除权重完成！-------"

echo "-------程序结束！！！-------"