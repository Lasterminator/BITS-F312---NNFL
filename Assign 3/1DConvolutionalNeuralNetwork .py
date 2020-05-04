import scipy
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D
import tensorflow as tf

def load_dataset():
    mat_X = scipy.io.loadmat('data_for_cnn.mat')['ecg_in_window']
    data_X = np.array(mat_X)
    mat_y = scipy.io.loadmat('y.mat')['label']
    data_y = np.array(mat_y)
    data = np.append(data_X,data_y,axis = 1) 
    np.random.shuffle(data)
    X = data[0:800, :-1]
    y = data[0:800, -1]
    X_test = data[800:1000, :-1]
    y_test = data[800:1000, -1]
    
    X = np.expand_dims(X,axis=2)
    y = np.expand_dims(y,axis=2)
    X_test = np.expand_dims(X_test,axis=2)
    y_test = np.expand_dims(y_test,axis=2)
    print (X.shape) 
    return X, y, X_test, y_test

def evaluatemodel(X, y, X_test, y_test):
    k = 3
    epochs = 20
    model = Sequential()
    model.add(Convolution1D(filters = 32,kernel_size = k,activation = 'relu',input_shape = (1000,1)))
    model.add(MaxPooling1D(2))
    model.add(Convolution1D(filters = 64,kernel_size = k,activation = 'relu'))
    model.add(MaxPooling1D(2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    print(model.summary)

    model.compile(loss = 'mse',optimizer='rmsprop',metrics=['accuracy'])
    history = model.fit(X,y,batch_size=2,epochs=epochs)
    print(history)

    loss,accuracy =model.evaluate(X_test,y_test,batch_size=2)
    
    return loss,accuracy

def run_experiments(repeats):
    X, y, X_test, y_test = load_dataset()
    scores = list()
    losses = list()
    for r in range(repeats):
        loss,score = evaluatemodel(X,y,X_test,y_test)
        score = score*100.0
        loss = loss*100.0
        scores.append(score)
        losses.append(loss)
    print(scores,losses)