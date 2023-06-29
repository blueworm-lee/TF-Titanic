import pandas as pd
import os

# 파일을 읽어온다.
data_origin_dir = os.path.join(os.getcwd(), "data", "origin")
data_modify_dir = os.path.join(os.getcwd(), "data", "modify")

ds_train = pd.read_csv(os.path.join(data_origin_dir, "train.csv"))
ds_test = pd.read_csv(os.path.join(data_origin_dir, "test.csv"))
ds_test_result = pd.read_csv(os.path.join(data_origin_dir, "test_result.csv"))

# Sex Set
ds_train['Sex'] = ds_train['Sex'].map({'male': 1, 'female': 2})
ds_test['Sex'] = ds_test['Sex'].map({'male': 1, 'female': 2})


# Embarked Set
ds_train['Embarked'] = ds_train['Embarked'].fillna('S')

# Embarked Set One-Hot
ds_train['Embarked'] = ds_train['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
ds_test['Embarked'] = ds_test['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

# Sir Set
def SetSir(row):
    return row.split(',')[1].split('.')[0]

ds_train['Sir'] = ds_train['Name'].apply(SetSir).str.strip()
ds_test['Sir'] = ds_train['Name'].apply(SetSir).str.strip()

# Age Set
ds_train['Age'] = ds_train['Age'].fillna(ds_train.groupby('Sir')['Age'].transform('mean'))
ds_test['Age'] = ds_test['Age'].fillna(ds_test.groupby('Sir')['Age'].transform('mean'))

# Age Range Set
def SetAgeRange(row):
    if row < 5:
        return 0
    elif row >= 5 and row <10:
        return 1
    elif row >= 10 and row <15:
        return 2
    elif row >= 15 and row <20:
        return 3
    elif row >= 20 and row <25:
        return 4
    elif row >= 25 and row <30:
        return 5
    elif row >= 30 and row <35:
        return 6
    elif row >= 35 and row <40:
        return 7
    elif row >= 40 and row <45:
        return 8
    elif row >= 45 and row <50:
        return 9
    elif row >= 50 and row <55:
        return 10
    elif row >= 55 and row <60:
        return 11
    elif row >= 60 and row <65:
        return 12
    elif row >= 65 and row <70:
        return 13
    elif row >= 70 and row <75:
        return 14
    elif row >= 75 and row <80:
        return 15
    else:
        return 16
    
ds_train['AgeRange'] = ds_train['Age'].apply(SetAgeRange)
ds_test['AgeRange'] = ds_test['Age'].apply(SetAgeRange)

# Fare Range Set
def SetFareRange(row):
    if row < 7.9:
        return 0
    elif row >= 7.9 and row <10.5:
        return 1
    elif row >= 10.5 and row <21.7:
        return 2
    elif row >= 21.7 and row <39.7:
        return 3
    else:
        return 4

ds_train['FareRange'] = ds_train['Fare'].apply(SetFareRange)
ds_test['FareRange'] = ds_test['Fare'].apply(SetFareRange)


# Family Set
ds_train['Family'] = 1+ds_train['SibSp'] + ds_train['Parch']
ds_test['Family'] = 1+ds_test['SibSp'] + ds_test['Parch']

# Alone Set
ds_train.loc[(ds_train['Family'] != 1), 'Alone'] = 0
ds_train.loc[(ds_train['Family'] == 1), 'Alone'] = 1

ds_test.loc[(ds_test['Family'] != 1), 'Alone'] = 0
ds_test.loc[(ds_test['Family'] == 1), 'Alone'] = 1

# Save to new feature set
ds_train.to_csv(os.path.join(data_modify_dir, "train_m.csv"))
ds_test.to_csv(os.path.join(data_modify_dir, "test_m.csv"))




