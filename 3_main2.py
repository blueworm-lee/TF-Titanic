import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def loss_graph(hist):
    plt.title('Loss')
    plt.ylabel('epochs')
    plt.ylabel('loss')
    plt.grid()

    plt.plot(hist.history['loss'], label='train loss')
    plt.plot(hist.history['val_loss'], label='validation loss')

    plt.legend(loc='best')
    plt.show()

def accuracy_graph(hist):
    plt.title('Accuracy')
    plt.ylabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()

    plt.plot(hist.history['accuracy'], label='train accuracy')
    plt.plot(hist.history['val_accuracy'], label='validation accuracy')

    plt.legend(loc='best')
    plt.show()



# Get Data
data_origin_dir = os.path.join(os.getcwd(), "data", "origin")
data_modify_dir = os.path.join(os.getcwd(), "data", "modify")

ds_train = pd.read_csv(os.path.join(data_modify_dir, "train_m.csv"))
ds_test = pd.read_csv(os.path.join(data_modify_dir, "test_m.csv"))
ds_test_result = pd.read_csv(os.path.join(data_origin_dir, "test_result.csv"))

#Feature Set
result_column = ['Survived']
feature_column = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'AgeRange', 'FareRange', 'Family', 'Alone']

train_feature_data = ds_train[feature_column]
train_result_data = ds_train[result_column]

test_feature_data = ds_test[feature_column]

#print(train_feature_data.shape, test_feature_data.shape)
#print(train_feature_data)
#print(train_result_data)
#print(test_feature_data)

# 정규화
def standard(train, test):
    mean = train.mean(axis=0)
    train -= mean
    std = train.std(axis=0)
    train /= std
    test -= mean
    test /= std
    

standard(train_feature_data, test_feature_data)
print(train_feature_data.head(3))
print(test_feature_data.head(3))

# Model Set
def my_model():
    
    x = tf.keras.layers.Input(shape=[len(feature_column)])    
    h = tf.keras.layers.Dense(128, activation='swish')(x)
    y = tf.keras.layers.Dense(2, activation='softmax')(h)
    
    '''
    h = tf.keras.layers.Dense(128, activation='swish')(x)
    h = tf.keras.layers.Dropout(0.2)(h)
    h = tf.keras.layers.Dense(64, activation='swish')(h)
    h = tf.keras.layers.Dropout(0.2)(h)
    y = tf.keras.layers.Dense(2, activation='softmax')(h)
    '''

    model = tf.keras.models.Model(x, y)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy', metrics='accuracy')

    return model

# Pandas class to numpy
x_train = train_feature_data.to_numpy().astype('float32')
x_test = test_feature_data.to_numpy().astype('float32')
y_train = train_result_data.to_numpy().astype('float32')

###################

# Devide Training and Validation Set 
cnt=4
num_val_samples = len(x_train) // cnt

print(num_val_samples)
model = my_model()

for i in range(cnt):
    print(f"#{i}번째 폴드 처리 시작")
    val_x = x_train[i*num_val_samples: (i+1)* num_val_samples]
    val_y = y_train[i*num_val_samples: (i+1)* num_val_samples]

    #print(len(val_x), len(val_y))
    partial_x = np.concatenate([x_train[:i*num_val_samples], x_train[(i+1)*num_val_samples:]], axis=0)
    partial_y = np.concatenate([y_train[:i*num_val_samples], y_train[(i+1)*num_val_samples:]], axis=0)

    #print(len(partial_x), len(partial_y))

    model.fit(partial_x, partial_y, epochs=50, validation_data=(val_x, val_y), verbose=1)
    #model.fit(partial_x, partial_y, epochs=100, verbose=1)


# Training
model.fit(partial_x, partial_y, epochs=200, verbose=1)

predict = model.predict(test_feature_data)

survived = np.argmax(predict, axis=1)

# to send kaggle
ds_test_result['Survived'] = np.argmax(predict, axis=1)
ds_test_result.to_csv(os.path.join(data_modify_dir, "result_to_kaggle.csv"), index=False)

#to_check result one more
ds_test_result['Result'] = np.argmax(predict, axis=1)
ds_test_result.to_csv(os.path.join(data_modify_dir, "test_result_m.csv"), index=False)

