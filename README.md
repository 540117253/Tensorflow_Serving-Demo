# Tensorflow_Serving-Demo

## 内容提要
==本文主要按照以下几点展开：==
1. `TensorFlow_Serving`的安装
2. `TensorFlow模型`的训练与保存
3. 启动`TensorFlow_Serving`加载模型来提供服务
4. 编写客户端，调用gRPC接口访问模型


==运行环境：==
- Tensorflow-1.13.1(可以比这更低的版本)
- Python-3.5

==整体的项目代码：==
- [GitHub传送门](https://github.com/540117253/Tensorflow_Serving-Demo)



## 1. 安装TensorFlow_Serving的docker镜像
首次使用docker时，每次都要敲`sudo docker...`才能执行docker。为了能直接运行docker指令，请先运行以下指令：
```
#添加docker用户组
sudo groupadd docker

#将登陆用户加入到docker用户组中
sudo gpasswd -a $USER docker

#更新用户组
newgrp docker     

#测试docker命令是否可以使用sudo正常使用
docker ps
```

参考[Tensorflow_Serving的官方Github文档](https://github.com/tensorflow/serving)，我运行以下指令安装`TensorFlow_Serving`。

首先新建一个名为`TensorFlow_Serving`的文件夹，并在该文件夹下执行：
```
# 下载 TensorFlow Serving Docker 镜像
docker pull tensorflow/serving

# 克隆tensorflow_serving的github代码到当前文件夹路径下
git clone https://github.com/tensorflow/serving

# demo模型的所在路径：在刚克隆的github代码中，$(pwd)代表当前路径
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"

# 开启 TensorFlow Serving container 和 REST API 端口
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &

# 使用 predict API 来访问模型
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict

# 安装成功的话，会出现以下结果
# Returns => { "predictions": [2.5, 3.0, 4.5] }

```

## 2.训练模型拟合`$ y = x^2+b $`, 并保存为Tensorflow_Serving能读取的格式
在该例子中，我们构造一个满足一元二次函数 y = x^2+b的训练数据。然后构建一个最简单的神经网络来训练出模型的参数。
```python
import tensorflow as tf
import numpy as np

'''
    1. 生成训练数据
'''
# 根据方程 y =x^2 + b，构建 x 和 y
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

'''
    2. Tensorflow 模型定义
'''
# 定义模型的输入
input_x = tf.placeholder(tf.float32, [None, 1]) # shape=[None,1]
input_y = tf.placeholder(tf.float32, [None, 1]) # shape=[None,1]

# 定义模型的中间层
L1 = tf.keras.layers.Dense(20, activation='relu')(input_x) # shape=[None,20]

# 定义模型的输出，其为根据输入的 x 而预测出的 y
pre_y = tf.keras.layers.Dense(1)(L1) # shape=[None,1]

# 定义模型损失函数，其为 预测的y 与 输入的y 的差值
diff = tf.subtract(pre_y, input_y)
loss = tf.nn.l2_loss(diff)

# 定义最小化目标函数的梯度优化器
train_step = tf.train.AdamOptimizer(learning_rate = 0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)


'''
    3. 训练模型
'''
init = tf.global_variables_initializer() # 初始化所有变量
sess = tf.Session()
sess.run(init)
# 训练 1000 次
for i in range(1000): 
    # 每一轮都根据输入的x和y，使用梯度优化器来最小化目标函数。每次运行后，都梯度下降来更新模型参数
    sess.run(train_step, feed_dict={input_x: x_data, input_y: y_data})
    # 每 50 次打印出一次损失值
    if i % 50 == 0: 
        print(sess.run(loss, feed_dict={input_x: x_data, input_y: y_data}))

'''
    4. 保存模型，其中要定义通过Tensorflow_Serving调用该模型的模型名称（这里我定义为“MMModel”），及进行预测操作时的方法名称（这里我定义为“your_prediction”）
'''
model_output_path = 'trained_models'
model_Name = 'MMModel'
model_Version = '1'

# 定义模型存储路径。这里存储路径的分层结构是tensorflow_serving所必须的。注意：tensorflow_serving默认加载 `model_Version`数值最大的模型版本
builder = tf.saved_model.builder.SavedModelBuilder(model_output_path +'/'+ model_Name+'/'+model_Version)

# 定义模型的输入与输出 到tf.saved_model
tensor_info_input_x = tf.saved_model.utils.build_tensor_info(input_x)
tensor_info_input_y = tf.saved_model.utils.build_tensor_info(input_y)
tensor_info_pre_y = tf.saved_model.utils.build_tensor_info(pre_y)

# 定义保存模型的核心参数设置，其包括模型的输入输出、调用该模型预测功能时的方法名称
prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
                'tensor_info_input_x': tensor_info_input_x,
                'tensor_info_input_y': tensor_info_input_y
                },
        outputs={'tensor_info_pre_y': tensor_info_pre_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

# 这里将调用模型预测功能的方法名称设置为 "your_prediction"
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
          'your_prediction': prediction_signature,
        },
        main_op =legacy_init_op
)

builder.save()

```
保存成功后，在当前目录下能看到多出一个准备用于`Tensorflow_Serving`加载模型的目录`trained_models`，其目录结构如下：
- trained_models
    - MMModel
        - 1

## 3. 使用Tensorflow_Serving加载训练好的模型，并通过gRPC接口访问模型
为了顺利使用gRPC接口，这里首先需要安装`tensorflow-serving-api`
```
pip install tensorflow-serving-api -i https://pypi.tuna.tsinghua.edu.cn/simple
```

接下来启动`Tensorflow_Serving`加载训练好的模型（注意以下指令需要在刚从github克隆下来的`serving`底下执行）：

```python
'''
    1. 将本机的8700端口映射到docker中的8500端口. 第二个端口必须为8500，因为tensorflow_serving服务端口在docker镜像里面为8500
    2. 将本机中的source路径映射到docker中的target路径，其中source为保存模型的(!绝对路径!)，target为docker中的路径。
    3. MODEL_NAME为提供服务的模型名称（客户端访问时需要填写正确）
'''
docker run -p 8700:8500 --mount type=bind,source=/home/Review-based-Collaborative-Filtering/trained_models/MMModel,target=/models/MMModel -e MODEL_NAME=MMModel -t tensorflow/serving &
```

最后编写客户端代码，发送模型的输入数据至`Tensorflow_Serving`的指定模型，得出预测结果：
```
'''python
    客户端连接`Tensorflow_Serving`，使用指定模型进行预测的代码
'''

import tensorflow as tf
import grpc
import numpy as np
from tensorflow_serving.apis import model_service_pb2_grpc, model_management_pb2, get_model_status_pb2, predict_pb2, prediction_service_pb2_grpc
from tensorflow_serving.config import model_server_config_pb2
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.core.framework import types_pb2


# 根据方程 y =x^2 + b，构建模型输入的 x 和 y
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


# 连接Tensorflow_Serving服务器
channel = grpc.insecure_channel('localhost:8700')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


# 定义所连接的模型名称及操作名称
request = predict_pb2.PredictRequest()
request.model_spec.name = 'MMModel' # 对应 Tensorflow_Serving启动服务时所设置的模型名称
request.model_spec.signature_name = 'your_prediction' # 对应模型保存时所设置的操作名称

# 构建请求报文
request.inputs['tensor_info_input_x'].CopyFrom(
  tf.contrib.util.make_tensor_proto(x_data, shape=x_data.shape, dtype=types_pb2.DT_FLOAT))
request.inputs['tensor_info_input_y'].CopyFrom(
  tf.contrib.util.make_tensor_proto(y_data, shape=y_data.shape, dtype=types_pb2.DT_FLOAT))

# 发送请求报文并接收预测结构
result = stub.Predict(request)

''' 
顺利访问Tensorflow_Serving后，返回模型预测的result的结构如下：
outputs {
  key: "tensor_info_pre_y"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 300
      }
      dim {
        size: 1
      }
    }
    float_val: 0.4377797245979309
    float_val: 0.4279484748840332
            ........
    float_val: 0.4181172251701355
  }
}
model_spec {
  name: "MMModel"
  version {
    value: 1
  }
  signature_name: "your_prediction"
}
'''

# 提取模型的预测结果，可以使用：
result.outputs['tensor_info_pre_y'].float_val

```



