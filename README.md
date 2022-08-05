# DeepCTR-tf2
> tf2学习过程中练习用的代码，集中在常见的ctr预估算法

## 基本的文件结构
- DeepCTR-tf2
  - src：主要代码的存放路径
    - custom_utils：自定义工具类，主要包括自定义layers、losses、metrics、callback等
    - datas：数据读取工具类，解析原始数据的各种工具类
    - models：模型实现
      - functional
      - sequential
      - subclassing
    - utils：一些其他的工具
  - examples：测试数据的存放路径
    - conf：配置文件
    - datas：训练、测试数据
  - docs: 文档文件
    - pics: 文档内容里使用到的图片

## 测试用例

### 1. 模型声明

* 调用说明：
```python
import tensorflow as tf

from src.utils.config import Config
from src.models.subclassing import mlp

config_ = Config('../examples/conf/')

mlp_model = mlp.MLPBuilder(config=config_)
tf.keras.utils.plot_model(mlp_model.build_graph(), show_shapes=True)
```
* 输出结果：
![img.png](docs/pics/img.png)

### 2. 数据读取
```python
# 数据读取测试
import os
from src.datas.load_data import CustomDataLoader

data_dir = './examples/datas/train.txt'

batch_size = config_.read_data_batch_size()
epochs = config_.read_data_epochs()

data_loader = CustomDataLoader(
    config_, mode='train',
    data_path=os.path.join(data_dir, 'train.txt')
    ).input_fn(batch_size_=batch_size, epochs_=epochs)

```

### 3. 模型训练和评估
```python
import tensorflow as tf
from src.utils.runner import Runner

# 模型runner初始化
model_runner = Runner(mlp_model, config_)

# 进行模型训练
model_runner.run(data_loader)

# 模型评估
model_runner.run(data_loader, steps=50, training=False)
```
* 评估指标数据：
> the batch 50 total loss: 54.00279235839844
> 
> the metrics: 0.7162307500839233