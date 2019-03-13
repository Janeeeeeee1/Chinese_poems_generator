import datetime
import tensorflow as tf

import os

import Preprocess
from BatchGenerator import BatchGenerator
from Model import Model

word2int, poem_vectors, int2word= Preprocess.build_dataset()

empty_key = word2int[' ']

batch_size =64

batch_generator = BatchGenerator(poem_vectors, batch_size, empty_key)

input_size = output_size = len(word2int) + 1
#定义参数
#因为每个batch的长度不一样，所以用None自动调整长度,这里的None为num_steps
train_data = tf.placeholder(tf.int32, [batch_size, None])
train_label = tf.placeholder(tf.int32, [batch_size, None])

model = Model(X=train_data, batch_size=batch_size, input_size=input_size, output_size=output_size)

logits, last_state, _, _ = model.print_variables()
#把train_label转化成[batch_size*num_steps]的一维向量
targets = tf.reshape(train_label, [-1])
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],  #[batch_size*num_steps, output_size]
                                                          [targets], #[batch_size * num_steps]
                                                          [tf.ones_like(targets, dtype=tf.float32)], len(word2int))
# 对损失求平均
cost = tf.reduce_mean(loss)
# 定义global_step,不对global_step更新
global_step = tf.Variable(0, trainable=False)

decay_step=batch_generator._batch_num
decay_rate = 0.96
learning_rate = tf.train.exponential_decay(0.01, global_step, decay_step, decay_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(cost))
# 防止参数膨胀，若大于5则设置为5
gradients, _ = tf.clip_by_global_norm(gradients, 5)
optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    print("training...")
    model_dir = "./model/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print("create the directory: %s" % model_dir)
    # 损失值最小的回合
    best_cost_epoch = 0
    # 损失最小值
    best_cost = float('Inf')
    start_time = datetime.datetime.now()
    for epoch in range(101):
        epoch_start_time = datetime.datetime.now()
        epoch_mean_cost = 0
        for batch in range(batch_generator._batch_num):
            x_data, y_data = batch_generator.next()
            _, _, c, lr, gs = session.run(
                [optimizer, last_state, cost, learning_rate, global_step],
                feed_dict={train_data: x_data, train_label: y_data})
            epoch_mean_cost += c
            print("current epoch %d, current batch is %d, mean cost : %2.8f, learning rate: %2.8f, global step : %d"
                %(epoch, batch, c, lr, gs))
        epoch_mean_cost = epoch_mean_cost / batch_generator._batch_num
        print("="*80)
        if epoch != 0:
            print("\nthe best cost : %2.8f, the best epoch index : %d, current epoch cost : %2.8f. \n" \
                %(best_cost, best_cost_epoch, epoch_mean_cost))
        if best_cost > epoch_mean_cost:
            print("the best epoch will change from %d to %d" %(best_cost_epoch, epoch))
            best_cost = epoch_mean_cost
            best_cost_epoch = epoch
            saver.save(session, model_dir + 'poetry.module-best')
        if epoch % 7 == 0:
            saver.save(session, model_dir + 'poetry.module', global_step=epoch)
        end_time = datetime.datetime.now()
        timedelta = end_time - epoch_start_time
        print("the epoch training spends %d days, %d hours, %d minutes, %d seconds.\n" \
            %(timedelta.days, timedelta.seconds // 3600, timedelta.seconds // 60, timedelta.seconds % 60))
        print("="*80)
    print("\n")
    timedelta = end_time - start_time
    print("*"*80)
    print("\nThe training spends %d days, %d hours, %d minutes, %d seconds" \
            %(timedelta.days, timedelta.seconds // 3600, timedelta.seconds // 60, timedelta.seconds % 60))
