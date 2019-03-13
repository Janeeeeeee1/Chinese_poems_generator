import numpy as np
import tensorflow as tf

import Preprocess
from Model import Model

word2int, poem_vectors, int2word = Preprocess.build_dataset()

def to_word(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    return int2word[sample]

# 定义输入的只有一个字词，然后根据上一个字词推测下一个词的位置
input_data = tf.placeholder(tf.int32, [1, None])
# 输入和输出的尺寸为1
input_size = output_size = len(int2word) + 1
# 定义模型
model = Model(X=input_data, batch_size=1, input_size=input_size, output_size=output_size)
# 获取模型的输出参数
_, last_state, probs, initial_state = model.print_variables()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    print("generate...")
    saver.restore(session, './model/poetry.module-best')
    # 起始字符是'['，
    x = np.array([list(map(word2int.get, '['))])
    # 运行初始0状态
    state_ = session.run(initial_state)
    word = poem = '['
    # 结束字符是']'
    while word != ']':
        # 使用上一级的state和output作为输入
        probs_, state_ = session.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        word = to_word(probs_)
        poem += word
        # 获取词语的id
        x = np.zeros((1, 1))
        x[0, 0] = word2int[word]
    print(poem)
