import os
import sys

import lightgbm as lgb
import pandas as pd
import shap
from sklearn import datasets as ds


def plot_print_feature_shap(model_path, data_feats, it_type):
    """
    利用shap打印特征重要度
    :param model_path:
    :param data_feats:
    :param it_type:
    :return:
    """

    if not (os.path.exists(model_path) and os.path.exists(data_feats)):
        print("file no exists! {}, {}".format(model_path, data_feats))
        sys.exit(0)
    gbm = lgb.Booster(model_file=model_path)
    gbm.params["objective"] = "regression"
    # feature列名
    feats_col_name = []
    for feat_index in range(46):
        feats_col_name.append('feat' + str(feat_index) + 'name')
    X_train, _ = ds.load_svmlight_file(data_feats)
    # features
    feature_mat = X_train.todense()
    df_feature = pd.DataFrame(feature_mat)
    # 增加表头
    df_feature.columns = feats_col_name
    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(df_feature[feats_col_name])

    # 特征总体分析，分别绘出散点图和条状图
    if it_type == 1:
        # 把一个特征对目标变量影响程度的绝对值的均值作为这个特征的重要性(不同于feature_importance的计算方式)
        shap.summary_plot(shap_values, df_feature[feats_col_name], plot_type="bar")
        # 对特征总体分析
        shap.summary_plot(shap_values, df_feature[feats_col_name])
    # 部分依赖图的功能，与传统的部分依赖图不同的是，这里纵坐标不是目标变量y的数值而是SHAP值
    if it_type == 2:
        shap.dependence_plot('feat3name', shap_values, df_feature[feats_col_name], interaction_index=None, show=True)
    # 两个变量交互下变量对目标值的影响
    if it_type == 3:
        shap.dependence_plot('feat3name', shap_values, df_feature[feats_col_name], interaction_index='feat5name',
                             show=True)
    # 多个变量的交互进行分析
    if it_type == 4:
        shap_interaction_values = explainer.shap_interaction_values(df_feature[feats_col_name])
        shap.summary_plot(shap_interaction_values, df_feature[feats_col_name], max_display=4, show=True)


def plot_print_feature_importance(model_path):
    """
    打印特征的重要度
    :param model_path:
    :return:
    """

    # 模型中的特征是Column_数字,这里打印重要度时可以映射到真实的特征名
    # feats_dict = {
    #     'Column_0': '特征0名称',
    #     'Column_1': '特征1名称',
    #     'Column_2': '特征2名称',
    #     'Column_3': '特征3名称',
    #     'Column_4': '特征4名称',
    #     'Column_5': '特征5名称',
    #     'Column_6': '特征6名称',
    #     'Column_7': '特征7名称',
    #     'Column_8': '特征8名称',
    #     'Column_9': '特征9名称',
    #     'Column_10': '特征10名称',
    # }
    feats_dict = {}
    for feat_index in range(46):
        col = 'Column_' + str(feat_index)
        feats_dict[col] = 'feat' + str(feat_index) + 'name'

    if not os.path.exists(model_path):
        print("file no exists! {}".format(model_path))
        sys.exit(0)

    gbm = lgb.Booster(model_file=model_path)

    # 打印和保存特征重要度
    fea_importance = gbm.feature_importance(importance_type='split')
    feature_names = gbm.feature_name()

    sum_im = 0.
    for value in fea_importance:
        sum_im += value

    for feature_name, importance in zip(feature_names, fea_importance):
        if importance != 0:
            feat_id = int(feature_name.split('_')[1]) + 1
            print('{} : {} : {} : {}'.format(feat_id, feats_dict[feature_name], importance, importance / sum_im))


def plot_tree(model_path, tree_index, save_plot_path):
    """
    对模型进行可视化
    :param model_path:
    :param tree_index:
    :param save_plot_path:
    :return:
    """
    if not os.path.exists(model_path):
        print("file no exists! {}".format(model_path))
        sys.exit(0)
    gbm = lgb.Booster(model_file=model_path)
    graph = lgb.create_tree_digraph(gbm, tree_index=tree_index, name='tree' + str(tree_index))
    graph.render(filename=save_plot_path, view=True)  # 可视图保存到save_plot_path中
