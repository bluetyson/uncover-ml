# from lightgbm.sklearn import LGBMModel
#
# class XGBoost(LGBMModel):
#     def __init__(self):
#         super(self).__init__()
#
# if __name__ == '__main__':
#     print('main ran')


# import lightgbm as lgb
# import pandas as pd
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import GridSearchCV
#
# # load or create your dataset
# print('Load data...')
# df_train = pd.read_csv('./regression.train', header=None, sep='\t')
# df_test = pd.read_csv('./regression.test', header=None, sep='\t')
#
# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values
#
# print('Start training...')
# # train
# gbm = lgb.LGBMRegressor(objective='regression',
#                         num_leaves=31,
#                         learning_rate=0.05,
#                         n_estimators=20)
# gbm.fit(X_train, y_train,
#         eval_set=[(X_test, y_test)],
#         eval_metric='l1',
#         early_stopping_rounds=5)
#
# print('Start predicting...')
# # predict
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# # eval
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
#
# print('Calculate feature importances...')
# # feature importances
# print('Feature importances:', list(gbm.feature_importances_))
#
# # other scikit-learn modules
# estimator = lgb.LGBMRegressor(num_leaves=31)
#
# param_grid = {
#     'learning_rate': [0.01, 0.1, 1],
#     'n_estimators': [20, 40]
# }
#
# gbm = GridSearchCV(estimator, param_grid)
#
# gbm.fit(X_train, y_train)
#
# print('Best parameters found by grid search are:', gbm.best_params_)
class ff:
    def __init__(self):
        self.c=[]

def fff(i):
    i.append(5)

b = 2
fff(b)
print(b)

a = ff()
fff(a.c)
print(a.c)