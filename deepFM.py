import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras import optimizers
from keras.engine.topology import Layer

# 样本和标签，这里需要对应自己的样本做处理
train_x = [
    np.array([0.5, 0.7, 0.9]),
    np.array([2, 4, 6]),
    np.array([[0, 1, 0, 0, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1, 0, 1],
              [0, 1, 0, 0, 0, 1, 0, 1]])
]
label = np.array([0, 1, 0])

# 输入定义
continuous = Input(shape=(1, ), name='single_continuous')
single_discrete = Input(shape=(1, ), name='single_discrete')
multi_discrete = Input(shape=(8, ), name='multi_discrete')

# FM 一次项部分
continuous_dense = Dense(1)(continuous)
single_embedding = Reshape([1])(Embedding(10, 1)(single_discrete))
multi_dense = Dense(1)(multi_discrete)
# 一次项求和
first_order_sum = Add()([continuous_dense, single_embedding, multi_dense])

# FM 二次项部分 k=3
continuous_k = Dense(3)(continuous)
single_k = Reshape([3])(Embedding(10, 3)(single_discrete))
multi_k = Dense(3)(multi_discrete)
# 先相加后平方
sum_square_layer = Lambda(lambda x: x**2)(
    Add()([continuous_k, single_k, multi_k]))
# 先平方后相加
continuous_square = Lambda(lambda x: x**2)(continuous_k)
single_square = Lambda(lambda x: x**2)(single_k)
multi_square = Lambda(lambda x: x**2)(multi_k)
square_sum_layer = Add()([continuous_square, single_square, multi_square])

substract_layer = Lambda(lambda x: x * 0.5)(
    Subtract()([sum_square_layer, square_sum_layer]))


# 定义求和层
class SumLayer(Layer):
  def __init__(self, **kwargs):
    super(SumLayer, self).__init__(**kwargs)

  def call(self, inputs):
    inputs = K.expand_dims(inputs)
    return K.sum(inputs, axis=1)

  def compute_output_shape(self, input_shape):
    return tuple([input_shape[0], 1])


# 二次项求和
second_order_sum = SumLayer()(substract_layer)
# FM 部分输出
fm_output = Add()([first_order_sum, second_order_sum])

# deep 部分
deep_input = Concatenate()([continuous_k, single_k, multi_k])
deep_layer_0 = Dropout(0.5)(Dense(64, activation='relu')(deep_input))
deep_layer_1 = Dropout(0.5)(Dense(64, activation='relu')(deep_layer_0))
deep_layer_2 = Dropout(0.5)(Dense(64, activation='relu')(deep_layer_1))
deep_output = Dropout(0.5)(Dense(1, activation='relu')(deep_layer_2))

concat_layer = Concatenate()([fm_output, deep_output])
y = Dense(1, activation='sigmoid')(concat_layer)

model = Model(inputs=[continuous, single_discrete, multi_discrete], outputs=y)

Opt = optimizers.Adam(lr=0.001,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=None,
                      decay=0.0,
                      amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=Opt, metrics=['acc'])
model.fit(train_x,
          label,
          shuffle=True,
          epochs=1,
          verbose=1,
          batch_size=1024,
          validation_split=None)
