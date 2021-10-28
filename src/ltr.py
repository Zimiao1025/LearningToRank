import lightgbm as lgb
import numpy as np

def train(x_train, y_train, q_train, model_save_path):
    """
    模型的训练和保存
    :param x_train:
    :param y_train:
    :param q_train:
    :param model_save_path:
    :return:
    """

    train_data = lgb.Dataset(x_train, label=y_train, group=q_train)
    params = {
        'task': 'train',  # 执行的任务类型
        'boosting_type': 'gbrt',  # 基学习器
        'objective': 'lambdarank',  # 排序任务(目标函数)
        'metric': 'ndcg',  # 度量的指标(评估函数)
        'max_position': 10,  # @NDCG 位置优化
        'metric_freq': 1,  # 每隔多少次输出一次度量结果
        'train_metric': True,  # 训练时就输出度量结果
        'ndcg_at': [10],
        'max_bin': 255,  # 一个整数，表示最大的桶的数量。默认值为 255。lightgbm 会根据它来自动压缩内存。如max_bin=255 时，则lightgbm 将使用uint8 来表示特征的每一个值。
        'num_iterations': 200,  # 迭代次数，即生成的树的棵数
        'learning_rate': 0.01,  # 学习率
        'num_leaves': 31,  # 叶子数
        # 'max_depth':6,
        'tree_learner': 'serial',  # 用于并行学习，‘serial’： 单台机器的tree learner
        'min_data_in_leaf': 30,  # 一个叶子节点上包含的最少样本数量
        'verbose': 2  # 显示训练时的信息
    }
    gbm = lgb.train(params, train_data, valid_sets=[train_data])
    gbm.save_model(model_save_path)


def predict(x_test, comments, model_input_path):
    """
     预测得分并排序
    :param x_test:
    :param comments:
    :param model_input_path:
    :return:
    """

    gbm = lgb.Booster(model_file=model_input_path)  # 加载model

    ypred = gbm.predict(x_test)

    predicted_sorted_indexes = np.argsort(ypred)[::-1]  # 返回从大到小的索引

    t_results = comments[predicted_sorted_indexes]  # 返回对应的comments,从大到小的排序

    return t_results
