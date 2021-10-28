import lightgbm as lgb

from data_format_read import read_dataset
from ndcg import validate


def test_data_ndcg(model_path, test_path):
    """
    评估测试数据的ndcg
    :param model_path:
    :param test_path:
    :return:
    """

    with open(test_path, 'r', encoding='utf-8') as testfile:
        test_X, test_y, test_qids, comments = read_dataset(testfile)

    gbm = lgb.Booster(model_file=model_path)
    test_predict = gbm.predict(test_X)

    average_ndcg, _ = validate(test_qids, test_y, test_predict, 60)
    # 所有qid的平均ndcg
    print("all qid average ndcg: ", average_ndcg)
    print("job done!")
