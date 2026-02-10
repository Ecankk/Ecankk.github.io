---
title: Task01：Al时代，会说话就会编程
author: Ecank
tags:
  - LLM
  - DataWhale
created: 2026-02-10 23:14
completed?: false
keyword_for_dataview: ""
share: true
category: docs/ML/DataWhale/vibe coding入门
modify: 2026-02-11 00:14
---
# vibe coding 入门
现在 AI 的发展为我们的开发带来了极大的便利，使用 AI 来编写代码对于任何人来说都是非常容易的事情。对于计算机出身的人来说，掌握所有的技术栈仍然不是非常的现实，但是通过 AI 可以很大程度上弥补这一点，特别是自身在课业上的使用情况来说，非常轻量级的 demo 或者说是一个简单独立的功能，AI 基本上可以完成的非常好。但是如果尝试让 AI 来完成一个完整的 project 而言，面临的苦难就很多，尤其是涉及较多的内容，对于 AI 而言并不是能轻易的完成。
这就要求人去把握项目的进度，通过认为的功能拆解，让 AI 完成独立的部分，从而避免因为项目过大而造成的问题。当好一个合格的产品经理去使用 AI。如果能做到拆分的细致，基本上不需要人为的看代码和修正，从而避免 AI 全局把控能力不足从而造成的拆东墙补西墙。（因为我经常让 AI 修改一个 bug 后造成了更多的 bug）。
# vibe coding 尝试

## prompt
使用 [z.ai](https://chat.z.ai/)
```
帮我做一个贪吃蛇游戏：
1. 用方向键控制蛇的移动
2. 吃到食物后蛇会变长，分数增加
3. 撞到墙壁或自己的身体就游戏结束
4. 要有开始和重新开始按钮
5. 界面要简洁好看
```
## 效果 1
![image.png|900](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20260210233327.png)
经过测试后修复了一些小 bug 后的截图效果
![image.png|900](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20260210233822.png)

对于一个网页的小游戏来说，完成度已经是可以了。
## 不一样的提示词
```
 **示例提示词：** 帮我做一个贪吃蛇游戏，它应该支持：

1. 我可以吃不同的单词，它们会被收集在一个盒子里
2. 当蛇吃了8个单词时，llm 应该根据这些单词创作一首诗，我们可以根据需要重新混合这首诗。
3. 当诗完成后，下一步将自动根据这首诗创建一幅图像。
```
## 效果 2
经过了一些和 AI 的 debug
![image.png|900](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20260211000313.png)
![image.png|900](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20260211000320.png)
![image.png|900](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/20260211000328.png)

可能就是网页化的编程带来的 AI 味道非常重（这里的 AI 味主要是模板的一致和界面的风格），但是从一个小游戏的角度来说，效果很不错了。

# 其他的 vibe coding 结果
这是我之前尝试利用本地tts 配合 LLM api 做交互的虚拟偶像交流网页项目 [CyberIdol](https://github.com/Ecankk/CyberIdol_Project)，支持文本/语音输入后，选择音色（可导入），利用 prompt 设置来选择态度从而实现不用的态度回答和交互。同时允许认识 prompt 的设定，主要的代码工作由 gemeni 和 codex 来完成。
![1.png|900](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/1.png)
![2.png|900](https://eeecank-1325470508.cos.ap-shanghai.myqcloud.com/2.png)
