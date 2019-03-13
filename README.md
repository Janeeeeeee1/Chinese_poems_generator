# Chinese_poems_generator

* tensorflow-gpu 1.13.1
* python 3.6.7
* gpu rtx2070mq

## 概述
使用了大约3.5万首诗进行训练，训练了100个epoches
运行Train.py进行训练，我是用gpu训练的所以速度还是比较快的，模型的参数保存在model文件夹下
在运行Write.py时使用的是cost最小的参数。
以下是运行Write.py产生的一些结果：



## 思考：
* 数据量不够大，如果想要更大的数据量，可以在https://github.com/chinese-poetry/chinese-poetry
中的json文件夹中找到大约30万首诗（包括宋朝的诗和唐朝的诗，不过是繁体字的），可以使用json2txt.py文件将json格式的数据转换成 title:content的形式，进行训练，效果会好很多。
* 平仄没有学到，中国古诗最重要的一点是平仄，但是平仄很难被模型学到。
* 意思不通，embedding是用tf.random_normal随机生成的，但如果使用已经训练好的词向量应该会有更好的效果。
* 生成的诗基本上都是不知所云，如果能根据给定的意象生成一首诗，会有更好的结果，但至今没有想到很好的解决办法。
