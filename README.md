# 问答摘要与推理项目
question and answer summary and reasoning

训练集（82943条记录）建立模型，基于汽车品牌、车系、问题内容与问答对话的文本，输出建议报告文本。

测试集（20000条记录）使用训练好的模型，输出建议报告的结果文件。
# 项目文件说明
seq2seq_pgn_tf2文件下是使用tensorflow2.0搭建完成的两个模型，一个是baseline版本的seq2seq模型，另外一个是基于seq2seq的Pointer-Generator Networks（PGN）模型。

seq2seq_paddle是使用paddlepaddle搭建的模型（暂停开发）

# 代码部分
## seq2seq_pgn_tf2文件下baseline版本的seq2seq模型
1. preprocess.py

完成原始数据的解析与存储

2. data_reader.py

读取数据，并建立vocab

3. utils/build_w2v.py

利用word2vector方法预训练词向量

4. main.py

完成模型的训练和预测
## seq2seq_pgn_tf2文件下PGN模型
程序执行顺序参照上一个模型

# 优秀作业分享
## HCT一期
tf2版本的PGN模型

https://github.com/shellrazer/Project_1_kaikeba

## HCT二期
tf2版本的PGN模型

https://github.com/Light2077/QA-Abstract-And-Reasoning

paddlepaddle版本的seq2seq

https://github.com/Abner1zhou/Q-A-Summary-and-Reasoning

pytorch版本的PGN模型

https://github.com/425776024/summary

https://github.com/quanterk/pgn_summarization_for_baidu_competition