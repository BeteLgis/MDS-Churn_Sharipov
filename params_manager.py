import json
import uuid
import random
import os


from shutil import rmtree
from pathlib import Path
from functools import wraps
from datetime import timedelta
from timeit import default_timer as timer

from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np
import matplotlib.pyplot as plt


PATH = Path('.')
DATASETS_PATH = PATH / 'data'
INTERNAL_PATH = PATH / Path('internal')
PARAMS_PATH = PATH / Path('params')


def seed_everything(seed):
    # fix all types of random that we can
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def timeit(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        if kwargs.pop('no_timer', False):
            return func(*args, **kwargs)

        start = timer()
        result = func(*args, **kwargs)
        runtime = timer() - start
        if kwargs.pop('get_time', False):
            return runtime
        print(f"\'{func.__name__}\' function runtime: {timedelta(seconds=int(round(runtime, 0)))}")
        return result
    return wrap


def get_column_names(df, include_time=False, full=False):
    target_column = 'target'

    dates = ['report_dt', 'max_date__agg_user', 'min_date__agg_user', 'first_account_hit',
             'first_account_charge', 'last_account_hit', 'last_account_charge', ]
    ignore = ['user_id', 'mode_weekday__agg_user', 'mode_month__agg_user', 'mode_time_of_day__agg_user',
              'mode_season_of_year__agg_user', 'mode_mcc_code__agg_user', ] + dates

    cat_features = ['most_season_of_year__agg_user', 'most_time_of_day__agg_user', 'employee_count_nm']

    num_features = list(set(df.columns) - set(ignore) -
                        set(cat_features) - {'time', 'target'})

    ignored_features = ['user_id']

    feature_columns = list(set(num_features) | set(cat_features) | set(ignored_features))

    if include_time:
        feature_columns += ['predict_time', 'time_pred_001', 'time_pred_005',
                            'time_pred_01', 'time_pred_03', 'time_pred_05', 'time_pred_07']
        num_features += ['predict_time', 'time_pred_001', 'time_pred_005',
                         'time_pred_01', 'time_pred_03', 'time_pred_05', 'time_pred_07']

    if full:
        return feature_columns, target_column, num_features, cat_features, ignored_features

    return feature_columns, target_column


def save_params(to_save, model_name, delete_old=False, params_path=PARAMS_PATH):
    Path(params_path).mkdir(parents=True, exist_ok=True)

    if not isinstance(to_save, list):
        to_save = [to_save]
    if len(to_save) > 0:
        path = params_path / model_name
        Path(path).mkdir(parents=True, exist_ok=True)

        if delete_old:
            rmtree(path, ignore_errors=True)
            Path(path).mkdir(parents=True, exist_ok=True)

        for params in to_save:
            with open(path / f'{uuid.uuid4().hex}.json', 'w', encoding='utf-8') as f:
                json.dump(params, f)


def load_params(model_name, PARAMS_DIR=PARAMS_PATH):
    paths = sorted(Path(PARAMS_DIR / model_name).iterdir(), key=os.path.getmtime)
    res = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            res.append(json.load(f))
    return res


def plot_roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)

    plt.figure(figsize=(10, 3))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.4f)' % roc_auc, alpha=0.5)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver operating characteristic', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()
    return roc_auc
