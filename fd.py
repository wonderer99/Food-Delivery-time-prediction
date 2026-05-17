import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error ,r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')
df.columns = df.columns.str.lower()
'''
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df.dtypes)

'''
def preprocess(df):
    df  = df.copy()
    
    df = df.drop(['id','delivery_person_id','order_date'],axis=1,errors='ignore')
    
    if 'time_taken(min)' in df.columns:
       df['time_taken(min)'] = df['time_taken(min)'].str.extract(r'(\d+)').astype(float)
    
    df['delivery_person_age'] = pd.to_numeric(df['delivery_person_age'],errors='coerce')
    df['delivery_person_ratings'] = pd.to_numeric(df['delivery_person_ratings'],errors='coerce')
    df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'],errors='coerce')
    
    df['distance'] = np.sqrt(df['delivery_location_latitude'] - df['restaurant_latitude']**2 
                             + (df['delivery_location_longitude']- df['restaurant_longitude']**2))
    
    df = df.dropna(subset=['time_orderd', 'time_order_picked'])

    df['time_orderd'] = pd.to_datetime(df['time_orderd'], format='%H:%M:%S', errors='coerce')
    df['time_order_picked'] = pd.to_datetime(df['time_order_picked'], format='%H:%M:%S', errors='coerce')

    
    df = df.dropna(subset=['time_orderd', 'time_order_picked'])
    
    df['cook_diff'] = (df['time_order_picked']  - df['time_orderd']).dt.seconds/60
    
    df['order_hour'] = df['time_orderd'].dt.hour
    
    df = df.drop(['delivery_location_latitude','delivery_location_longitude','restaurant_latitude','restaurant_longitude','time_orderd','time_order_picked'],axis=1)
    
    cat_cols = ['weatherconditions','road_traffic_density','type_of_order','type_of_vehicle','festival','city']
    df = pd.get_dummies(df, columns = cat_cols,drop_first=True)
    df.columns = df.columns.str.strip()
    bool_cols = df.select_dtypes('bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    df = df.fillna(df.median(numeric_only=True))
        
    return df

df_clean = preprocess(df)
print(df_clean.head())
print(df_clean.dtypes)
print(df_clean.isnull().sum())

X = df_clean.drop('time_taken(min)',axis=1)
y = df_clean['time_taken(min)']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators = 100,
    learning_rate = 0.05,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8,
    early_stopping_rounds = 10,
    random_state = 42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test,y_test)],
    verbose=False
)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test,preds)
rmse = np.sqrt(mean_squared_error(y_test,preds))
r2 = r2_score(y_test,preds)

print(f"mae : {mae:.2f} minutes")
print(f"rmse: {rmse:.2f} minutes")
print(f"r2 : {r2:.2f}")

model.save_model('models/food_delivery_time_model.json')
print("Model saved successfully.")

test_df = pd.read_csv('data/test.csv')
test_df.columns = test_df.columns.str.lower()



test_clean = preprocess(test_df)

test_ids = test_clean['id'] if 'id' in test_clean.columns else test_df.loc[test_clean.index, 'id']


test_clean_model = test_clean.reindex(columns=X.columns, fill_value=0)

test_preds = model.predict(test_clean)

submission = pd.DataFrame({ 
    'id': test_ids,
    'time_taken(min)': test_preds                           
                           })

submission.to_csv('data/submission.csv', index=False)
print("Submission file created successfully")
print(submission.head())

xgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# correlation of every feature with delivery time
correlation = df_clean.corr(numeric_only=True)['time_taken(min)'].sort_values(ascending=False)
print(correlation)