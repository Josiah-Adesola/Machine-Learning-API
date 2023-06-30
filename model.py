import pandas as pd
import numpy as np

url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
df = pd.read_csv(url)
include = ['Age', 'Sex', 'Embarked', 'Survived']
df_= df[include]

# This handles non-categorical observations and replace them with 0
# While it  put allthe categorical values in a list


categoricals = []
for col, col_type in df_.dtypes.items():
    if col_type == 'O':
        categoricals.append(col)
    else:
        df_.loc[:,col].fillna(0, inplace=True)
        
df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

from sklearn.linear_model import LogisticRegression
dependent_variable = 'Survived'
X = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]

lr = LogisticRegression()
lr.fit(X,y)

# save the model to a pickle file
import joblib
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')



    