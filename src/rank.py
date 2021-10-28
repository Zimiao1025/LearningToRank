import os
import sys
from datetime import datetime

from preprocess import process_data_format, load_data, load_data_from_raw
from ltr import train, predict
from plot import plot_tree, plot_print_feature_shap, plot_print_feature_importance
from evaluation import test_data_ndcg


def rank_func():
    if len(sys.argv) != 2:
        print("Usage: python lgb_ltr.py [-process | -train | |-plot | -predict | -ndcg | -feature | -shap | -leaf]")
        sys.exit(0)

    base_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    train_path = base_path + '/data/train/'
    raw_data_path = train_path + 'raw_train.txt'
    data_feats = train_path + 'feats.txt'
    data_group = train_path + 'group.txt'

    model_path = base_path + '/data/model/model.mod'
    save_plot_path = base_path + '/data/plot/tree_plot'

    if sys.argv[1] == '-process':
        # 训练样本的格式与ranklib中的训练样本是一样的,但是这里需要处理成lightgbm中排序所需的格式
        # lightgbm中是将样本特征feats和groups分开保存为txt的,什么意思呢,看下面解释
        '''
        输入：
        1 qid:0 1:0.2 2:0.4 ... #comment
        2 qid:0 1:0.1 2:0.2 ... #comment
        1 qid:1 1:0.2 2:0.1 ... #comment
        3 qid:1 1:0.3 2:0.7 ... #comment
        2 qid:1 1:0.5 2:0.5 ... #comment
        1 qid:1 1:0.6 2:0.3 ... #comment

        输出：
        feats:
        1 1:0.2 2:0.4 ...
        2 1:0.1 2:0.2 ...
        1 1:0.2 2:0.1 ...
        3 1:0.3 2:0.7 ...
        2 1:0.5 2:0.5 ...
        1 1:0.6 2:0.3 ...
        groups:
        2
        4

        以上group中2表示前2个是一个qid,4表示后4个是一个qid

        '''
        process_data_format(raw_data_path, data_feats, data_group)

    elif sys.argv[1] == '-train':
        # train
        train_start = datetime.now()
        x_train, y_train, q_train = load_data(data_feats, data_group)
        train(x_train, y_train, q_train, model_path)
        train_end = datetime.now()
        consume_time = (train_end - train_start).seconds
        print("consume time : {}".format(consume_time))

    elif sys.argv[1] == '-plot_tree':
        # 可视化树模型
        plot_tree(model_path, 2, save_plot_path)

    elif sys.argv[1] == '-predict':
        train_start = datetime.now()
        # predict_data_path = base_path + '/data/test/test.txt'  # 格式如ranklib中的数据格式
        test_X, test_y, test_qids, comments = load_data_from_raw(raw_data_path)
        t_results = predict(test_X, comments, model_path)
        print(t_results)
        train_end = datetime.now()
        consume_time = (train_end - train_start).seconds
        print("consume time : {}".format(consume_time))

    elif sys.argv[1] == '-ndcg':
        # ndcg
        test_path = base_path + '/data/test/test.txt'  # 评估测试数据的平均ndcg
        test_data_ndcg(model_path, test_path)

    elif sys.argv[1] == '-feature':
        plot_print_feature_importance(model_path)

    elif sys.argv[1] == '-shap':
        plot_print_feature_shap(model_path, data_feats, 3)
