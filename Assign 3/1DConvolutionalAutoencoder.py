import scipy
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling2D, GlobalAveragePooling1D, Conv2D,UpSampling2D
import tensorflow as tf
from keras.engine.input_layer import Input
from keras.optimizers import RMSprop

def load_dataset():
    mat_X = scipy.io.loadmat('data_for_cnn.mat')['ecg_in_window']
    X = np.array(mat_X)
    np.random.shuffle(X)
    X = np.expand_dims(X,axis=2)
    X_test = np.expand_dims(X_test,axis=2)
    print(X.shape[0]) 
    print(X.shape)
    return X

def autoencoder(input_X):
    print(input_X.shape)
    # encoding part
    conv1 = Conv2D(32,(5,1),activation='relu',padding='same')(input_X)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = Conv2D(64,(5,1),activation='relu',padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = Conv2D(128,(5,1),activation='relu',padding='same')(pool2)
    # decoding part
    conv4 = Conv2D(128,(5,1),activation='relu',padding='same')(conv3)
    up1 = UpSampling2D((2,2))(conv4)
    conv5 = Conv2D(128,(5,1),activation='relu',padding='same')(conv2)
    up2 = UpSampling2D((2,2))(conv5)
    decoded = Conv2D(1,(5,1),activation='sigmoid',padding='same')(up2)
    return decoded

X = Input(shape=(100,10,1))
autoencoder = Model(X,autoencoder(X))
autoencoder.compile(loss='mean_squared_error',optimizer=RMSprop())
print(autoencoder.summary)

X = load_dataset()
train_data = X.reshape(1000,100,10,1)
train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,train_data,test_size = 0.2, random_state=13)
autoencoder_train= autoencoder.fit(train_X,train_ground,batch_size=2,epochs=250,verbose=1,validation_data=(valid_X,valid_ground))
print(train_X.shape)

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(250)
plt.figure()
plt.plot(epochs)
plt.title('Training loss')
plt.legend()
plt.show()