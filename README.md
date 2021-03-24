# 航空旅客付费选座意愿识别
### 赛题要求
本项目为2021年软件服务外包大赛《航空旅客付费选座意愿识别》赛题，该赛题要求从650个特征因子中，提取或衍生出能够描述旅客付费选座意愿的特征因子集合；并利用特征因子集合的成果，构建旅客付费选座意愿识别模型。    

### 模型架构
由于题目给定的特征非常多，我们采用PNN(Product-based-Neural-Network)深度学习模型对特征工程进行模型化。模型原始论文在项目目录data下。该模型的Embedding层能够对类别型特征组成的稀疏矩阵转换为大量稠密矩阵，提高模型收敛速率，同时Product层能够将特征域的Embedding后的向量进行两两交叉，充分挖掘组合特征，获取特征之间的交叉信息。    

### 环境配置
Python (3.6)  
numpy (1.19.5)  
pandas (1.1.5)  
scikit-learn (0.24.1)  
tensorflow==2.0.0b1  


### 数据集
数据集直接采用的赛方给定的旅客付费选座数据集，放在项目目录中的data文件夹下。  


### 项目运行
运行:
```
python3 main.py
```
