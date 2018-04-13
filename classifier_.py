from keras import backend as K
K.set_image_dim_ordering('th')
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
##


batch_size = 32
nb_classes = 2
nb_epoch = 150

#datadir = "/home/shubhabrata/Desktop/CookPad"
"""
X_train = np.load(join(datadir, 'X_train.npy'))
X_train_ = X_train.transpose((0, 3, 1, 2))
X_test = np.load(join(datadir, 'X_test.npy'))
X_test_ = X_test.transpose((0, 3, 1, 2))
y_train = np.load(join(datadir, 'y_train.npy'))
y_test = np.load(join(datadir, 'y_test.npy'))
"""
X_train = np.load('X_train.npy')
X_train_ = X_train.transpose((0, 3, 1, 2))
X_test = np.load('X_test.npy')
X_test_ = X_test.transpose((0, 3, 1, 2))
y_train = np.load('y_train.npy')
y_test = np.load( 'y_test.npy')


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

"""
# print one sample image
plt.imshow(X_train[2])

plt.imshow(X_train[12])
"""

# Data pre-processing and augmentation:

X_train_ = X_train_.astype('float32')
X_test_ = X_test_.astype('float32')
X_train_ /= 255
X_test_ /= 255


# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
    
# Compute quantities required for featurewise normalization (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train_)

# Define the network


model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train_.shape[1:], dim_ordering = 'th'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same',dim_ordering = 'th'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Convolution2D(128, 3, 3, border_mode='same',dim_ordering = 'th'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train_, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train_.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test_, Y_test))



TrueLabels = y_test
PredictedLabels = model.predict_classes(X_test_)

target_names = ['Sandwich', 'Sushi']
print(classification_report(TrueLabels, PredictedLabels, target_names=target_names))

print('Area under the curve (AUC) is:', roc_auc_score(TrueLabels, PredictedLabels))

"""
cm = confusion_matrix(PredictedLabels, TrueLabels)
fig, ax = plt.subplots()
im = ax.matshow(cm)
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
plt.title('Confusion matrix')
fig.colorbar(im)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
"""