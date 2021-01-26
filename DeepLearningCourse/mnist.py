import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as Linear
import numpy as np
import PIL as Image
trainset = paddle.dataset.mnist.train()
train_reader = paddle.batch(trainset,batch_size=8)

for batch_id,data in enumerate(train_reader()):
	img_data = np.array([x[0] for x in data]).astype('float32')
	label_data = np.array([x[1] for x in data]).astype('float32')
	print("图像数据开关和对应数据为：",img_data.shape,img_data[0])
	break

print("\n打印第一个batch的第一个图像，对应的标签数字为{}".format(label_data[0]))

import matplotlib.pyplot as plt
img = np.array(img_data[0] + 1) * 127.5
img = np.reshape(img,[28,28]).astype(np.uint8)
plt.figure("Image")
plt.axis('on')
plt.imshow(img)
plt.title('image')
plt.show()
