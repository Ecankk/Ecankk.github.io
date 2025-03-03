---
title: L0G1000 Linux 基础知识
author: Ecank
tags:
  - "#LLM"
created: 2025-03-01 20:27
updated: 
completed?: true
keyword_for_dataview: ""
share: true
category: docs/ML/InternLM
modify: 2025-03-03 18:14
---

# 闯关任务 
* 配置好 ssh 公钥后再 vscode 上连接上 InternStudio
	* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301203741.png)
* 建立 `helloworld.py`，并安装相应依赖
	* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301203851.png)

* 通过指令 `ssh -p ***** root@ssh.intern-ai.org.cn -CNg -L ，7860:127.0.0.1:7860 -o StrictHostKeyChecking=no` 建立端口映射，并运行程序，成功在本地端口访问
	* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301203955.png)
	* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301204030.png)
# 可选任务 1
尝试 Linux 相关命令
*  `touch` 创建文件
	* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301204437.png)
* `mkdir` 创建目录
	* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301204523.png)
* `cd` 访问目录
	* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301204551.png)
* `pwd` 显示目录
	* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301204638.png)
* `cat` 查看文件内容
	* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301204718.png)
* `rm` 和 `mv` 删除和移动/重命名
* `find` 查询符合条件的文件或目录
	* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301205858.png)
* `ls` 查看当前目录的内容和详细信息
	* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301205936.png)
# Conda 创建环境
* 尝试创建一个 conda 新环境
* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301210818.png)
* 创建成功并进入
* ![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250301211036.png)
