import tensorflow as tf

input_x = tf.constant([
    [[[5, 6, 0, 1, 8, 2],
      [0, 9, 8, 4, 6, 5],
      [2, 6, 5, 3, 8, 4],
      [6, 3, 4, 9, 1, 0],
      [7, 5, 9, 1, 6, 7],
      [2, 5, 9, 2, 3, 7]

      ]]])
filters = tf.constant([
    [[[0, -1, 1], [1, 0, 0], [0, -1, 1]]]
])

input_x=tf.reshape(input_x,(1,6,6,1))
filters=tf.reshape(filters,[3,3,1,1])

res = tf.nn.conv2d(input_x, filters, strides=1, padding='VALID')
print('Valid 无激活函数下的输出',res)
res=tf.squeeze(res)
print('Valid 条件下可视化的输出：',res)


# print('Valid 激活函数下输出',tf.nn.relu(res))
print('Valid 激活函数下可视化输出：',tf.squeeze(tf.nn.relu(res)))
#在full卷积下，TF中没有这个参数，可以手动加0实现
input_x = tf.constant([
    [[[0,0,0,0,0,0,0,0],
  [0,5,6,0,1,8,2,0],
  [0,2,5,7,2,3,7,0],
  [0,0,7,2,4,5,6,0],
  [0,5,3,6,9,3,1,0],
  [0,6,5,3,1,4,6,0],
  [0,5,2,4,0,8,7,0],
    [0,0,0,0,0,0,0,0]
]]])
input_x=tf.reshape(input_x,(1,8,8,1))

res = tf.nn.conv2d(input_x, filters, strides=1,padding='SAME')
print('Full（加0）未使用激活之前的输出',res)

print('Full(加0）未使用激活函数之前的可视化输出，',tf.squeeze(res))

out = tf.nn.relu(res)
print('Full 激活的输出',out)
print('Full 激活之后的可视化输出，',tf.squeeze(out))