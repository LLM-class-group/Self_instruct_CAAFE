#!/bin/bash

# 外层循环，重复若干次
for j in $(seq 1 1000)
do
  # 内层循环，运行多次main.py，每次处理一个数据集
  for i in $(seq 1 25)
  do
    # 忽略输出
    python caafe/main.py $i > /dev/null 2>&1
    echo -e "\033[1;32m\nCompleted dataset $i\n\033[0m"
  done
  echo -e "\033[1;32m\nCompleted iteration $j\n\033[0m"
done
