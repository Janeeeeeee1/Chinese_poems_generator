import tensorflow as tf
import Preprocess
class Model(object):
    """docstring for GeneratePoetryModel"""
    def __init__(self, X, batch_size, input_size, output_size, model='lstm', rnn_size=128, num_layers=2):
        self._model = model
        self._num_unit = rnn_size   # LSTM的单元个数
        self._num_layers = num_layers # LSTM的层数
        self._input_size = input_size
        self._output_size = output_size
        self._model_layers = self._get_layer()
        self._initial_state = self._model_layers.zero_state(batch_size, tf.float32)

        with tf.variable_scope('softmax'):
            n = (self._num_unit + self._output_size) * 0.5
            scale = tf.sqrt(3 / n)
            # 全连接层的参数定义
            softmax_w = tf.get_variable(
                "softmax_w",
                [self._num_unit, self._output_size],
                initializer=tf.random_uniform_initializer(-scale, scale))
            softmax_b = tf.get_variable(
                "softmax_b",
                [self._output_size],
                initializer=tf.random_uniform_initializer(-scale, scale))
            with tf.device("/gpu:0"):
                embedding = tf.get_variable("embedding", [self._input_size, self._num_unit])
                inputs = tf.nn.embedding_lookup(embedding, X)

        # LSTM
        outputs, last_state = tf.nn.dynamic_rnn(self._model_layers, inputs, initial_state=self._initial_state, scope="softmax")
        self._outputs = tf.reshape(outputs, [-1, self._num_unit])
        self._last_state = last_state
        # 得到全连接层结果
        self._logits = tf.matmul(self._outputs, softmax_w) + softmax_b
        # 得到预测结果
        self._probs = tf.nn.softmax(self._logits)

    def _get_cell(self):
        if self._model == 'rnn':
            cell_fun = tf.nn.rnn_cell.BasicRNNCell
        elif self._model == 'gru':
            cell_fun = tf.nn.rnn_cell.GRUCell
        elif self._model == 'lstm':
            cell_fun = tf.nn.rnn_cell.BasicLSTMCell

        return cell_fun(self._num_unit, state_is_tuple=True)

    def _get_layer(self):
        cell = self._get_cell()
        return tf.nn.rnn_cell.MultiRNNCell([cell] * self._num_layers, state_is_tuple=True)

    def print_variables(self):
        # 输出参数
        return self._logits, self._last_state, self._probs, self._initial_state