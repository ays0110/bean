from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.models import load_model
import h5py
from keras import optimizers
import numpy as np
import os


from sklearn import preprocessing

filters = [128, 64, 32]
kernel_size = [10, 5, 5]
pool_size = [2, 2, 3]
upsample_size = [4,3,2]
weight_decay = 0.00001

model = Sequential()
    ### Encoding portion
model.add(Conv1D(filters[0],
                     kernel_size[0],
                     input_shape=(300,1),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
#model.add(BatchNormalization(axis=-1, momentum=0.99))
model.add(MaxPooling1D(pool_size[0], padding='same'))

model.add(Conv1D(filters[1],
                     kernel_size[1],
                     activation='relu',
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
#model.add(BatchNormalization(axis=-1, momentum=0.99))
model.add(MaxPooling1D(pool_size[1], padding='same'))

model.add(Conv1D(filters[2],
                     kernel_size[2],
                     activation='relu',
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
#model.add(BatchNormalization(axis=-1, momentum=0.99))
model.add(MaxPooling1D(pool_size[2], padding='same'))



### Decoding portion
model.add(Conv1D(filters[1],
                     kernel_size[1],
                     activation='relu',
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
#model.add(BatchNormalization(axis=-1, momentum=0.99))
model.add(UpSampling1D(upsample_size[1]))


model.add(Conv1D(filters[0],
                     kernel_size[0],
                     activation='relu',
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
#model.add(BatchNormalization(axis=-1, momentum=0.99))
model.add(UpSampling1D(upsample_size[0]))


    # Decoded output
model.add(Conv1D(1, 1, activation='linear', padding='same'))


if __name__ == '__main__':
    data = np.load("../Data/fmri_data_2017_09_13.npy")
    #data = np.nan_to_num(data)
    data = np.transpose(data)
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    data = np.transpose(data)
    data = np.expand_dims(data, axis=2)

    random = np.random.permutation(np.shape(data)[0])
    ratio = .70
    number = int(ratio * np.shape(random)[0])
    print(number)
    training = data[0:number,:,:]
    test = data[number,:,:]


    print('Training data shape:', np.shape(training))
    print('Testing data shape:', np.shape(test))

    #sgd = optimizers.SGD(lr=0.001, clipvalue=0.5, momentum=.9, decay=0.0001)
    model.compile(optimizer='Adadelta', loss='mse')
    model.fit(training, training, batch_size=128, epochs=25, verbose=1, shuffle=True, validation_data=(test,test))

    model.save('../Analysis/models/25_epochs_pig_2_2017_09_13_(filter10_05_05).h5')