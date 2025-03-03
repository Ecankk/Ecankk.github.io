---
title: L0G2000 Python基础知识
author: Ecank
tags:
  - LLM
created: 2025-03-02 16:00
updated: 
completed?: true
keyword_for_dataview: ""
share: true
category: docs/ML/InternLM
modify: 2025-03-03 18:31
---

# Leetcode 383
```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        dicta=defaultdict(int)
        dictb=defaultdict(int)
        for i in ransomNote:
            dicta[i]+=1
        for i in magazine:
            dictb[i]+=1
        for i in ransomNote:
            if  dicta[i]>dictb[i]:
                return False
        return True
```
只要满足前一个字符中每一个字符的数量都小于等于后一个字符串中对应字符的数量就可以了
![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250302160048.png)
# Vscode连接InternStudio debug笔记
尝试直接运行程序，但是缺少openai 的module，所以先安装这个库，尝试运行，得到如图结果

![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250302201559.png)
![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250302201721.png)
经过 debug 打断点发现错误是在这里
![image.png|1000](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250302202034.png)
此时 res 的值是 `'根据提供的模型介绍文字，以下是提取的信息，以JSON格式返回：\n\n```json\n{\n  "model_name": "书生浦语InternLM2.5",\n  "development_institution": "上海人工智能实验室",\n  "parameter_versions": ["1.8B", "7B", "20B"],\n  "context_length": "1M"\n}\n```\n\n这个JSON对象包含了模型名字、开发机构、提供参数版本以及上下文长度这四个要求的信息。'`

说明此时返回的字符串不符合JSON 的解析标准
首先优化 prompt 为，并删除返回的多余字符
![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250302203021.png)
然后就能成功运行并输出结果了
![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250302203031.png)
# pip安装到指定目录
安装 `numpy` 到指定目录，执行语句 `pip install -t /root/myenvs numpy`
编写测试程序
```python
import sys
sys.path.append("/root/myenvs")
try:
    import numpy as np
    print("numpy版本:", np.__version__)
except ImportError:
    print("错误: 无法导入numpy！")
```
![image.png|500](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20250302210431.png)
成功了