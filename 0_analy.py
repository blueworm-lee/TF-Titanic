import matplotlib.pyplot as plt
import pandas as pd
import os

#Survived : 0(사망), 1(생존)
#Pclass: 1(1등석)
#Sibsp : Sibling count
#Parch : Parent & Child Count
#Ticket: 티켓번호
#Fare  : 여객운임
#Cabin : 객실번호
#Embarked: S(Southampton) C(Cherbourg) Q(Queenstown)


# 파일을 읽어온다.
data_origin_dir = os.path.join(os.getcwd(), "data", "origin")

ds_train = pd.read_csv(os.path.join(data_origin_dir, "train.csv"))
ds_test = pd.read_csv(os.path.join(data_origin_dir, "test.csv"))
ds_test_result = pd.read_csv(os.path.join(data_origin_dir, "test_result.csv"))

#Feature list
columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


for idx in range(8):     
    plt.subplot(2,4, idx+1)

    plt.title(str(columns[idx]))
    ds_train[str(columns[idx])].hist()

plt.show()



print('성별 생존여부------------------------')
print(ds_train.query('Survived == 1').groupby('Sex').count())

print('나이/성별 생존여부------------------------')
agegrp = pd.cut(ds_train['Age'], bins=[0,10,20,30,40,50,60,70,80])
print(ds_train.pivot_table('Survived', ['Sex', agegrp]))


print('클래스별 나이/성별,Class별 생존여부------------------------')
agegrp = pd.cut(ds_train['Age'], bins=[0,18,40,80])
print(ds_train.pivot_table('Survived', ['Sex', agegrp], 'Pclass'))

print('성별 생존여부------------------------')
print(ds_train.info())

print('Null 인 사람 호칭------------------------')
ds_train['Sir'] = ds_train['Name'].str.extract('([A-Za-z]+)\.')
print(ds_train[ds_train['Age'].isna()].groupby('Sir').count())

print('사람 호칭------------------------')
print(ds_train.groupby('Sir').count())
