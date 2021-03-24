'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-03 14:51:59
LastEditors: ZhangHongYu
LastEditTime: 2021-03-24 08:49:27
'''
import numpy as np
import tensorflow as tf
from time import time
import math
from sklearn.metrics import roc_auc_score, f1_score
from tensorflow.keras.metrics import Accuracy
from DataReader import FeatureDictionary, DataParser
from decimal import Decimal

class PNN(tf.keras.Model):

    def __init__(self, feature_size, field_size,
                 embedding_size=8,
                 deep_layers=[32, 32], deep_init_size = 50,
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layer_activation=tf.nn.relu,
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss",
                 use_inner=True):
        super().__init__()
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size    # one-hot编码后的所有特征数
        self.field_size = field_size    # one-hot编码前的特征域数量
        self.embedding_size = embedding_size

        self.deep_layers = deep_layers       # 全连接层大小
        self.deep_init_size = deep_init_size # Product Layer大小
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layer_activation

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type

        self.use_inner = use_inner

        # init其他变量
        tf.random.set_seed(self.random_seed)

        # 这里的weights是一个字典，包括所有weight和bias
        self._weights = self._initialize_weights()

    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random.normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name='feature_embeddings'
        )
        weights['feature_bias'] = tf.Variable(
            tf.random.normal([self.feature_size, 1], 0.0, 1.0), name='feature_bias'
        )


        # Product Layers
        # 非线性
        if self.use_inner:
            weights['product-quadratic-inner'] = tf.Variable(
                tf.random.normal([self.deep_init_size, self.field_size],
                0.0, 0.01))
        else:
            weights['product-quadratic-outer'] = tf.Variable(
                tf.random.normal([self.deep_init_size, self.embedding_size, self.embedding_size],
                0.0, 0.01)
            )
        # 线性
        weights['product-linear'] = tf.Variable(tf.random.normal(
            [self.deep_init_size, self.field_size, self.embedding_size],
            0.0, 0.01))
        weights['product-bias'] = tf.Variable(tf.random.normal(
            [self.deep_init_size,],
            0.0, 1.0
        ))


        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.deep_init_size
        
        #先初始化第0个全连接层
        #glorot 权值初始化方法，U~( -sqrt(6/(din+dout)), +sqrt(6/(din + dout)))
        glorot = np.sqrt(6.0/(input_size + self.deep_layers[0]))
        weights['layer_0'] = tf.Variable(
            tf.random.uniform(
                [input_size, self.deep_layers[0]],
                -glorot, glorot
            )
        )
        weights['bias_0'] = tf.Variable(
            tf.random.uniform(
                [1, self.deep_layers[0]],
                -glorot, glorot
            )
        )
        # 依次递推地初始化最后一层之前的全连接层
        for i in range(1, num_layer):
            glorot = np.sqrt(6.0/(self.deep_layers[i - 1] + self.deep_layers[i]))
            weights['layer_%d' % i] = tf.Variable(
                tf.random.uniform(
                    [self.deep_layers[i-1], self.deep_layers[i]],
                    -glorot, glorot
                )
            )
            weights['bias_%d' % i] = tf.Variable(
                tf.random.uniform(
                    [1, self.deep_layers[i]],
                    -glorot, glorot
                )
            )
        # 初始化最后一层全连接层
        glorot = np.sqrt(5.0/(input_size + 1))
        weights['output'] = tf.Variable(
            tf.random.uniform(
                [self.deep_layers[-1], 1],
                -glorot, glorot
            )
        )
        weights['output_bias'] = tf.Variable(   
            tf.constant(
                0.01
            ),
            dtype = np.float32
        )

        return weights

    def call(self, feat_index, feat_value, label, dropout_deep):
        # Embeddings
        # self.weights['feature_embeddings]为(feature_size, embedding_size),(256 * 8),其中feature_size是把所有one-hot展开后的总维度
        # 而feat_index为=(batch_size, n_field=38，1), n_field为one-hot没展开时的域个数
        # 根据索引从W里面选n_field个被嵌入后的向量出来，最终得到(batch_size, n_field, 8)，38只是元素个数，实际上字典大小即元素取值范围为256

        # self.feat_index是embedding层的输入，也是对embedding矩阵的索引，比如输入index为3元素列表 ，也可以对W=(8, 2)进行索引
        # 其实就是从(8, 2)里面选3个出来二维向量出来，最终得到(3, 2)，feat_index每一个位置就是一个索引

        #所有特征域都进行嵌入，数值域也当做类型域进行处理
        embeddings = tf.nn.embedding_lookup(self._weights['feature_embeddings'], feat_index)
        # (batch_size, n_field, 1)
        feat_value = tf.reshape(feat_value, shape=[-1, self.field_size, 1])
        # 逐个元素相乘?何用?
        embeddings = tf.multiply(embeddings, feat_value) # N * F * K

        # Linear Singal
        linear_output = []
        for i in range(self.deep_init_size):
            linear_output.append(
                tf.reshape(
                    tf.reduce_sum(
                        tf.multiply(embeddings, self._weights['product-linear'][i]), 
                        axis=[1, 2]),
                        shape=(-1, 1)
                )
            )
        lz = tf.concat(linear_output, axis=1) # N * init_deep_size
  
        # Quardatic Signal
        quadratic_output = []
        if self.use_inner:
            for i in range(self.deep_init_size):
                theta = tf.multiply(
                    embeddings,
                    tf.reshape(
                        self._weights['product-quadratic-inner'][i],
                        (1, -1, 1)
                    )
                )   # N * F * K
                quadratic_output.append(
                    tf.reshape(
                        tf.norm(
                            tf.reduce_sum(
                                theta, axis=1
                            ),
                            axis=1
                        ),
                        shape=(-1, 1)
                    )
                )   # N * 1
        else:
            embedding_sum = tf.reduce_sum(embeddings, axis=1)
            p = tf.matmul(tf.expand_dims(embedding_sum, 2), tf.expand_dims(embedding_sum, 1)) #N * K * K
            for i in range(self.deep_init_size):
                theta = tf.multiply(p, tf.expand_dims(self._weights['product-quadratic-outer'][i], 0)) # N * K * K
                quadratic_output.append(
                    tf.reshape(
                        tf.reduce_sum(
                            theta,
                            axis=[1, 2]
                        ),
                        shape=(-1, 1)
                    )
                )   # N * 1
        
        lp = tf.concat(quadratic_output, axis=1) # N * init_deep_size

        y_deep = tf.nn.relu(
            tf.add(
                tf.add(lz, lp),
                self._weights['product-bias']
            )
        )
        y_deep = tf.nn.dropout(
            y_deep,
            dropout_deep[0]
        )

        # Deep component
        for i in range(0, len(self.deep_layers)):
            y_deep = tf.add(
                tf.matmul(
                    y_deep, self._weights["layer_%d" % i ]
                ),
                self._weights["bias_%d" % i]
            )
            y_deep = self.deep_layers_activation(y_deep)
            y_deep = tf.nn.dropout(y_deep, dropout_deep[i + 1])
        
        out = tf.add(
            tf.matmul(
                y_deep, self._weights['output']
            ),
            self._weights['output_bias']
        )

        if self.loss_type == 'logloss': #logloss为交叉熵的二类别情况 yp +(1-y)(1-p)        y=0,1
            out = tf.nn.sigmoid(out)    #二分类，此处用sigmoid，多分类是softmax
        return out


def evaluate_accuracy(data_iter, net):
    acc_sum , n = 0.0, 0
    f1_list = []
    f1_y, f1_y_hat, roc_auc_y, roc_auc_y_hat = [], [], [], [] # 样本不均衡可能导致在该折交叉验证中没有正例，计算不了auc
    for _, (Xi, Xv, y) in enumerate(data_iter):
        y_hat = net(Xi ,Xv, y, dropout_deep=[0] * len(net.dropout_dep))
        
        # 先用输出的概率计算auc
        roc_auc_y.append(y.numpy().reshape(-1, 1))
        roc_auc_y_hat.append(y_hat.numpy())
        
        y_hat = tf.cast(tf.logical_not(tf.less(y_hat, [0.5])), tf.int32)
        #此题输出y_hat就是p+无用，一般情况下对所有测试集样本的 I(argmax(p_label1, p_label2,...) == yi) 进行求和
        acc_sum += tf.reduce_sum(
            tf.cast(
                tf.equal(
                    # dropout_deep必须要在[0,1)之间
                    tf.reshape(y_hat, shape=(-1,)),
                    y
                ),
                dtype=tf.int32
            )
        ).numpy()

        f1_y.append(y.numpy().reshape(-1, 1))
        f1_y_hat.append(y_hat.numpy())

        n += y.shape[0]

    test_f1 =  f1_score(np.concatenate(tuple(f1_y), axis=0), np.concatenate(tuple(f1_y_hat), axis=0))
    # 样本不均衡，很可能没有正例，故这里计算auc很有可能出错
    test_roc_auc = roc_auc_score(np.concatenate(tuple(roc_auc_y), axis=0), np.concatenate(tuple(roc_auc_y_hat), axis=0))
    return acc_sum / n, test_f1, test_roc_auc


def train(net, train_iter, test_iter, loss_type, learning_rate, epochs, optimizer_type, batch_size):

    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        f1_y, f1_y_hat, roc_auc_y, roc_auc_y_hat = [], [], [], [] # 样本不均衡可能导致在该折交叉验证中没有正例，计算不了auc
        #所有样本的loss,　所有样本中预测正确的个数，所有样本总个数
        for Xi, Xv, y in train_iter:
            # print(y)
            with tf.GradientTape() as tape:
                y_hat = net(Xi, Xv, y , dropout_deep=net.dropout_dep) 
                # loss
                if loss_type == 'logloss': # 不用log_loss容易导致结果为0，后面算梯度的时候除0得none?
                    loss = tf.compat.v1.losses.log_loss(y, tf.reshape(y_hat, shape=(-1,))) 
                elif loss_type == "mse":
                    loss = tf.keras.losses.MSE(y, y_hat)
                loss = tf.reduce_sum(loss) #返回的是这个batch的loss向量，需要对其求和
            # unconnected_gradients为无法求导时返回的值，有none和zero可选择，默认为none
            # 这里建议用zero，否则后面grad/batch_size要报错
            grads = tape.gradient(loss, net.trainable_variables, unconnected_gradients="zero") 

            # optimizer
            if optimizer_type == 'sgd':
                optimizer = tf.keras.optimizers.SGD(learning_rate)
            elif optimizer_type == "adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8)
            elif optimizer_type == "adagrad":
                optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate,
                                                            initial_accumulator_value=1e-8)
            # tf.keras.optimizers.SGD 直接使用是随机梯度下降 theta(t+1) = theta(t) - learning_rate * gradient
            # 这里使用批量梯度下降，需要对梯度除以 batch_size, 相当于 optimizer.step(batch_size)
            optimizer.apply_gradients(zip([ tf.compat.v1.div(grad, batch_size) for grad in grads], net.trainable_variables))

            train_l_sum += loss.numpy()

            # 先用输出的概率计算auc
            roc_auc_y.append(y.numpy().reshape(-1, 1))
            roc_auc_y_hat.append(y_hat.numpy())

            # 以多分类为例，计算损失函数时用概率输出P(Y=0,1,2...|x;theta)组成的向量与y独热向量求交叉熵即可
            # 但算精度要离散化，要对一个batch的I(argmax(p_label1, p_label2,...) == yi)进行求和 
            # 二分类 >=0.5 预测为正类, <0.5　预测为负类
            y_hat = tf.cast(tf.logical_not(tf.less(y_hat, [0.5])), tf.int32)
            train_acc_sum += tf.reduce_sum(
                tf.cast(
                    tf.equal(
                        tf.reshape(y_hat, shape=(-1,)),
                        y
                    ), 
                    dtype=tf.int32
                )
            ).numpy() # 这里是二分类直接输出正例概率值, 一般情况下多分类对一个batch的 I(argmax(p_label1, p_label2,...) == yi) 进行求和

            f1_y.append(y.numpy().reshape(-1, 1))
            f1_y_hat.append(y_hat.numpy())
            
            n += y.shape[0] # n是总样本个数，这里累加n_batch
        # 训练集的相关评估指标
        train_loss = train_l_sum/n
        train_acc = train_acc_sum/n

        train_f1 =  f1_score(np.concatenate(tuple(f1_y), axis=0), np.concatenate(tuple(f1_y_hat), axis=0))
        # 样本不均衡，很可能没有正例，故这里计算auc很有可能出错
        train_roc_auc = roc_auc_score(np.concatenate(tuple(roc_auc_y), axis=0), np.concatenate(tuple(roc_auc_y_hat), axis=0))

        test_acc, test_f1, test_roc_auc = evaluate_accuracy(test_iter, net)

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, train f1 %.3f, test f1 %.3f, train auc %.3f, test auc %.3f' 
        % (epoch + 1, train_loss, train_acc, test_acc, train_f1, test_f1, train_roc_auc, test_roc_auc))


def k_fold_cross_valid(dfTrain, X_submission, folds, pnn_params, train_params):
    numeric_cols = []
    ignore_cols = []
    for col in dfTrain.columns:
        type_col = str(dfTrain[col].dtype)
        if (type_col == 'float32' or (type_col == 'int64' and col[:10]!='pref_month')):
            numeric_cols.append(col)

    fd = FeatureDictionary(dfTrain=dfTrain,
                           dfTest=X_submission,
                           numeric_cols=numeric_cols,
                           ignore_cols=ignore_cols)
    data_parser = DataParser(feat_dict= fd)

    # Xi_train ：列的序号
    # Xv_train ：列的对应的值
    # 这里不方便调用imblearn实现过采样，因为他不是直接存储为one-hot矩阵
    # 而是索引和值分开存储的。也就是说，要用tensorflow自带的embedding函数，
    # 就很难再调用imlearn中的过采样了
    Xi_train,Xv_train,y_train = data_parser.parse(df=dfTrain,has_label=True)
    Xi_submission,Xv_submission,ids_submission = data_parser.parse(df=X_submission)

    # print(y_train)
    pnn_params['feature_size'] = fd.feat_dim #包括one-hot所有维度的总维度,n_all_feature
    pnn_params['field_size'] = len(Xi_train[0]) #将one-hot看做整体的总的域个数, n_field

    _get = lambda x,l:[x[i] for i in l]



    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        pnn = PNN(**pnn_params)
        
        train_iter = tf.data.Dataset.from_tensor_slices((Xi_train_, Xv_train_, y_train_)).batch(train_params['batch_size'])
        test_iter = tf.data.Dataset.from_tensor_slices((Xi_valid_, Xv_valid_, y_valid_)).batch(train_params['batch_size'])

        train(
            pnn, train_iter, test_iter, **train_params)
