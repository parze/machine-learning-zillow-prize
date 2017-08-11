import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import random

random_state_seed = 42
random.seed(random_state_seed)


def prepare_data_all():
    print "\nReading original data ..."

    prop = pd.read_csv('../data/original/properties_2016.csv')
    label = pd.read_csv('../data/original/train_2016_v2.csv')

    print "\nPreprocessing data ..."

    for c, dtype in zip(prop.columns, prop.dtypes):
        if dtype == np.float64:
            prop[c] = prop[c].astype(np.float32)

    month = label['transactiondate'].map(lambda date: float(date[5:7]), na_action=None).to_frame()
    month.columns = ['month']
    label = pd.concat([label, month], axis=1)

    all_df = label.merge(prop, how='left', on='parcelid')

    ulimit = np.percentile(all_df.logerror.values, 99)
    llimit = np.percentile(all_df.logerror.values, 1)
    print "logerror upper limit: "+str(ulimit)
    print "logerror lower limit: "+str(llimit)

    all_df=all_df[ all_df.logerror > llimit ]
    all_df=all_df[ all_df.logerror < ulimit ]

    for c in all_df.dtypes[all_df.dtypes == object].index.values:
        all_df[c] = (all_df[c] == True)

    return all_df


def prepare_data_and_save():
    all_df = prepare_data_all()
    print "\nWrite prepared data to disk ..."
    all_df.to_csv('../data/all_df.csv')


def train_lightgbm(x, y):
    split = int(round(0.995 * x.shape[0]))
    if split == x.shape[0]:
        split = x.shape[0] - 5
    x_train, y_train, x_valid, y_valid = x[:split], y[:split], x[split:], y[split:]
    x_train = x_train.values.astype(np.float32, copy=False)
    x_valid = x_valid.values.astype(np.float32, copy=False)

    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_valid, label=y_valid)

    params = {
        'max_bin': 10,
        'learning_rate': 0.0021,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l1',
        'sub_feature': 0.5,
        'bagging_fraction': 0.85,
        'bagging_freq': 40,
        'num_leaves': 512,
        'min_data': 500,
        'min_hessian': 0.05
    }

    print "\nTraining model ..."
    clf = lgb.train(params, d_train, 400, d_valid)

    return clf


def train_xgboost(xgb_xy_train, y):
    print "\nTraining XGBoost ..."
    xgb_params = {
        'eta': 0.03295,
        'max_depth': 6,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': np.mean(y),
        'silent': 0
    }    
    return xgb.train(dict(xgb_params), xgb_xy_train, num_boost_round=200)


def linear_model_x(y_lightgbm, y_xgboost):
    return pd.DataFrame({
        'lightgbm': y_lightgbm,
        'xgboost': y_xgboost
    })


def train_linear_model(x_train_lm, y_train):
    model = linear_model.LinearRegression()
    model.fit(x_train_lm, y_train)
    return model


def predict_weights(weights, y_predict_baseline, y_predict_lightgbm, y_predict_xgboost):
    return weights[0]*y_predict_baseline + weights[1]*y_predict_lightgbm + weights[2]*y_predict_xgboost


def train_for_weights(y_predict_baseline, y_predict_lightgbm, y_predict_xgboost, y):
    print "\nTraning weights ... "
    best_weights = [0.3, 0.3, 0.4]
    best_mea = mean_absolute_error(y, predict_weights(best_weights, y_predict_baseline, y_predict_lightgbm, y_predict_xgboost))
    for x in range(0, 10000):
        w0 = random.random()
        w1 = (1 - w0)*random.random()
        weights = [w0, w1, 1 - w0 - w1]
        mea = mean_absolute_error(y, predict_weights(weights, y_predict_baseline, y_predict_lightgbm, y_predict_xgboost))
        if mea < best_mea:
            best_mea = mea
            best_weights = weights
    return best_weights


def print_mae(y_predict_baseline, y_predict_lightgbm, y_predict_xgboost, y_predict_weights, y):
    print "\nMean Absolute Error for "
    print "  Baseline: " + str(mean_absolute_error(y, y_predict_baseline))
    print "  LightGBM: " + str(mean_absolute_error(y, y_predict_lightgbm))
    print "  XGBoost:  " + str(mean_absolute_error(y, y_predict_xgboost))
    print "  Weighted: " + str(mean_absolute_error(y, y_predict_weights))


def train(all_df):
    x = all_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
    y = all_df['logerror'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=random_state_seed)
    y_train_mean = np.mean(y_train)

    #
    # training predictions
    #

    y_train_predict_baseline = np.ones(len(y_train))*y_train_mean

    lightgbm_model = train_lightgbm(x_train, y_train)
    y_train_predict_lightgbm = lightgbm_model.predict(x_train)

    xgb_xy_train = xgb.DMatrix(x_train, y_train)
    xgboost_model = train_xgboost(xgb_xy_train, y_train)
    y_train_predict_xgboost = xgboost_model.predict(xgb_xy_train)

    weights = train_for_weights(y_train_predict_baseline, y_train_predict_lightgbm, y_train_predict_xgboost, y_train)
    print "Weights: ", weights
    y_train_weight = predict_weights(weights, y_train_predict_baseline, y_train_predict_lightgbm, y_train_predict_xgboost)

    print_mae(y_train_predict_baseline, y_train_predict_lightgbm, y_train_predict_xgboost, y_train_weight, y_train)

    #
    # test predictions
    #

    y_test_predict_baseline = np.ones(len(y_test))*y_train_mean

    y_test_predict_lightgbm = lightgbm_model.predict(x_test)
    xgb_xy_test = xgb.DMatrix(x_test, y_test)
    y_test_predict_xgboost = xgboost_model.predict(xgb_xy_test)

    y_test_weight = predict_weights(weights, y_test_predict_baseline, y_test_predict_lightgbm, y_test_predict_xgboost)

    print_mae(y_test_predict_baseline, y_test_predict_lightgbm, y_test_predict_xgboost, y_test_weight, y_test)


#prepare_data_and_save()

#train(prepare_data_all())

train(all_df=pd.read_csv('../data/all_df.csv'))




