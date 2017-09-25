from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.models import load_model
import h5py
from keras import optimizers
import numpy as np
import os
import keras
from keras.models import load_model


from sklearn import preprocessing

data = np.load("../Data/fmri_preprocessed_2017_09_18.npy")
data = data.astype(float)
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
test = data[number:,:,:]


print('Training data shape:', np.shape(training))
print('Testing data shape:', np.shape(test))
filters = [128, 64, 32]
kernel_size = [40, 10, 5]
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

#sgd = optimizers.SGD(lr=0.001, clipvalue=0.5, momentum=.9, decay=0.0001)
model.compile(optimizer='Adadelta', loss='mse')
model.fit(training, training, batch_size=128, epochs=25, verbose=1, shuffle=True, validation_data=(test,test))

model.save('../Model/25_epochs_pig_2_2017_09_18_(filter40_10_05).h5')

first_layer_model = keras.models.Model(inputs=model.input, outputs=model.layers[5].output)
first_layer_activation_matrix = first_layer_model.predict(test)
np.save('25_epochs_pig_2_2017_09_18_(filter40_10_05)_activations', first_layer_activation_matrix)
