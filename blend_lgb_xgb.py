import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# 读取数据
train = pd.read_csv('../data/used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('../data/used_car_testB_20200421.csv', sep=' ')
all_data = pd.concat([train, test], ignore_index=True)

# 对 price 做 log1p
all_data['price'] = np.log1p(all_data['price'])

# 特征工程
all_data['power'] = all_data['power'].apply(lambda x: 600 if x > 600 else x)
all_data['reg_year'] = all_data['regDate'].apply(lambda x: int(str(x)[:4]))
all_data['reg_month'] = all_data['regDate'].apply(lambda x: int(str(x)[4:6]))
all_data['reg_day'] = all_data['regDate'].apply(lambda x: int(str(x)[6:]))
all_data['creat_year'] = all_data['creatDate'].apply(lambda x: int(str(x)[:4]))
all_data['creat_month'] = all_data['creatDate'].apply(lambda x: int(str(x)[4:6]))
all_data['creat_day'] = all_data['creatDate'].apply(lambda x: int(str(x)[6:]))

all_data['notRepairedDamage'] = all_data['notRepairedDamage'].apply(lambda x: 0 if x == '-' else 1)

all_data['power_bucket'] = pd.cut(all_data['power'], 10, labels=False)
new_cols = ['power_bucket', 'v_0', 'v_3', 'v_8', 'v_12']
for col1 in new_cols:
    for col2 in new_cols:
        if col1 != col2:
            all_data[f'{col1}_{col2}_sum'] = all_data[col1] + all_data[col2]
            all_data[f'{col1}_{col2}_diff'] = all_data[col1] - all_data[col2]

# 填充缺失值
for col in ['fuelType', 'gearbox', 'bodyType', 'model']:
    all_data[col] = all_data[col].fillna(0)

# 拆分数据
train_data = all_data[~all_data['price'].isnull()]
test_data = all_data[all_data['price'].isnull()]
X_train = train_data.drop(['SaleID', 'name', 'regDate', 'creatDate', 'price'], axis=1)
X_test = test_data.drop(['SaleID', 'name', 'regDate', 'creatDate', 'price'], axis=1)
y_train = train_data['price']

# 模型定义
lgb_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2021
)

xgb_model = XGBRegressor(
    max_depth=10,
    learning_rate=0.05,
    n_estimators=1000,
    gamma=0.005,
    subsample=0.9,
    colsample_bytree=0.7,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=2021,
    eval_metric='mae'
)

# 交叉验证 + 模型融合
skf = KFold(n_splits=5, shuffle=True, random_state=2021)
lgb_oof = np.zeros(len(X_train))
xgb_oof = np.zeros(len(X_train))
lgb_pred = np.zeros(len(X_test))
xgb_pred = np.zeros(len(X_test))

for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Training Fold {i+1}")
    tr_x, tr_y = X_train.iloc[train_idx], y_train.iloc[train_idx]
    vl_x, vl_y = X_train.iloc[val_idx], y_train.iloc[val_idx]

    # LightGBM
    lgb_model.fit(tr_x, tr_y, eval_set=[(vl_x, vl_y)])
    lgb_oof[val_idx] = lgb_model.predict(vl_x)
    lgb_pred += lgb_model.predict(X_test) / skf.n_splits

    # XGBoost
    xgb_model.fit(tr_x, tr_y, eval_set=[(vl_x, vl_y)])
    xgb_oof[val_idx] = xgb_model.predict(vl_x)
    xgb_pred += xgb_model.predict(X_test) / skf.n_splits

# 融合预测（权重可调）
oof_blend = 0.2 * lgb_oof + 0.8 * xgb_oof
test_blend = 0.2 * lgb_pred + 0.8 * xgb_pred

mae = mean_absolute_error(np.expm1(y_train), np.expm1(oof_blend))
print("融合后 MAE: {:.3f}".format(mae))

# 保存结果
submission = pd.DataFrame({'SaleID': test_data['SaleID'], 'price': np.expm1(test_blend)})
submission.to_csv('blend_submission.csv', index=False)
