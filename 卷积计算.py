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

res = tf.nn.conv2d(input_x, filters, strides=1, padding='SAME')
print('无激活函数下的输出',res)

print('激活函数下输出',tf.nn.relu(res))

'''
conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, 
       data_format="NHWC", dilations=[1, 1, 1, 1], name=None):

 input：输入的tensor，被卷积的图像，conv2d要求input必须是四维的。四个维度分别为[batch, in_height, in_width, in_channels]，即batch size，输入图像的高和宽以及单张图像的通道数。

 filter：卷积核，也要求是四维，[filter_height, filter_width, in_channels, out_channels]四个维度分别表示卷积核的高、宽，输入图像的通道数和卷积输出通道数。其中in_channels大小需要与 input 的in_channels一致。

strides：步长，即卷积核在与图像做卷积的过程中每次移动的距离，一般定义为[1，stride_h,stride_w,1]，stride_h与stride_w分别表示在高的方向和宽的方向的移动的步长，第一个1表示在batch上移动的步长，最后一个1表示在通道维度移动的步长，而目前tensorflow规定：strides[0] = strides[3] = 1，即不允许跳过bacth和通道，前面的动态图中的stride_h与stride_w均为1。

padding：边缘处理方式，值为“SAME” 和 “VALID”，熟悉图像卷积操作的朋友应该都熟悉这两种模式；由于卷积核是有尺寸的，当卷积核移动到边缘时，卷积核中的部分元素没有对应的像素值与之匹配。此时选择“SAME”模式，则在对应的位置补零，继续完成卷积运算，在strides为[1,1,1,1]的情况下，卷积操作前后图像尺寸不变即为“SAME”。
若选择 “VALID”模式，则在边缘处不进行卷积运算，若运算后图像的尺寸会变小。


'''


