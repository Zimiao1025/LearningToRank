from sklearn import datasets as ds
import numpy as np


def save_data(group_data, output_feature, output_group):
    """
    group与features分别进行保存
    :param group_data:
    :param output_feature:
    :param output_group:
    :return:
    """
    if len(group_data) == 0:
        return
    output_group.write(str(len(group_data)) + '\n')
    for data in group_data:
        # 只包含非零特征
        # feats = [p for p in data[2:] if float(p.split(":")[1]) != 0.0]
        feats = [p for p in data[2:]]
        output_feature.write(data[0] + ' ' + ' '.join(feats) + '\n')  # data[0] => level ; data[2:] => feats


def process_data_format(data_path, data_feats, data_group):
    """
    转为lightgbm需要的格式进行保存
    :param data_path:
    :param data_feats:
    :param data_group:
    :return:
    """

    with open(data_path, 'r', encoding='utf-8') as f_read:
        with open(data_feats, 'w', encoding='utf-8') as output_feature:
            with open(data_group, 'w', encoding='utf-8') as output_group:
                group_data = []
                group = ''
                for line in f_read:
                    if '#' in line:
                        line = line[:line.index('#')]
                    splits = line.strip().split()
                    if splits[1] != group:  # qid => splits[1]
                        save_data(group_data, output_feature, output_group)
                        group_data = []
                        group = splits[1]
                    group_data.append(splits)
                save_data(group_data, output_feature, output_group)


def load_data(feats, group):
    """
    加载数据
    分别加载feature,label,query
    :param feats:
    :param group:
    :return:
    """

    x_train, y_train = ds.load_svmlight_file(feats)
    q_train = np.loadtxt(group)

    return x_train, y_train, q_train
