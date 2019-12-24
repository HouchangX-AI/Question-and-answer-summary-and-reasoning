import logging
import os
import pathlib
import pandas as pd


root = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
TRAINSET_PATH = os.path.join(root, "datasets/AutoMaster_TrainSet.csv")
TESTSET_PATH = os.path.join(root, "datasets/AutoMaster_TestSet.csv")
TRAINSET_SEG_PATH = os.path.join(root, "datasets/train_set_seg.cs")
TESTSET_SEG_PATH = os.path.join(root, "datasets/test_set_seg.cs")


def get_logger(name, log_file=None):
    """
    logger
    :param name: 模块名称
    :param log_file: 日志文件，如无则输出到标准输出
    :return:
    """
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not log_file:
        handle = logging.StreamHandler()
    else:
        handle = logging.FileHandler(log_file)
    handle.setFormatter(format)
    logger = logging.getLogger(name)
    logger.addHandler(handle)
    logger.setLevel(logging.DEBUG)
    return logger


def read_datasets(train_path, test_path):
    train = pd.read_csv(train_path)
    print(train['Dialogue'])
    test = pd.read_csv(test_path)
    print(test.head())

    for k in ['Brand', 'Model', 'Question', 'Dialogue']:
        train[k] = train[k].apply(process_line)
        test[k] = test[k].apply(process_line)

    train['Report'] = train['Report'].apply(process_line)

    train.to_csv(TRAINSET_SEG_PATH, index=False, encoding='utf-8')
    test.to_csv(TESTSET_SEG_PATH, index=False, encoding='utf-8')