##

##


##



import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
##

os.environ["CUDA_VISIBLE_DEVICES"]="0";


##




import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar100
import numpy as np
import os
import time


##



from __future__ import print_function
import threading

def compute_gradient(weights_1,weights_2):
    gradient = []
    for i in range(len(weights_1)):
        inner_lst = []
        for j in range(len(weights_1[i])):
            inner_lst.append(weights_2[i][j] - weights_1[i][j])
        gradient.append(inner_lst)
    return gradient      
    
    
    
def update_thread(model, cache, gradient, lr, start, end):
    
    for i in range(start, end+1):
        print (i)
        new_weights = []
        for j in range(len(cache[i])):
            new_weights.append( cache[i][j] + lr * gradient[i][j]   )
        model.layers[i].set_weights(new_weights)
        
        
def update_with_gradient(model, cache, gradient, lr):
    weights = [layer.get_weights() for layer in model.layers]
  
    for i in range(len(weights)):

        new_weights = []

        for j in range(len(weights[i])):

            new_weights.append( cache[i][j] + lr * gradient[i][j]   )

        model.layers[i].set_weights(new_weights) 


##




##

batch_size = 256  ##

epochs = 200
data_augmentation = True
num_classes = 100
k = 6
##

subtract_pixel_mean = True

##

##

##

##

##

##

##

##

##

##

##

##

##

##

n = 18

##

##

version = 1

##

if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

##

model_type = 'ResNet%dv%d' % (depth, version)

##

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

##

input_shape = x_train.shape[1:]

##

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

##

if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

##

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
    


##




    

val = 0
def lr_schedule(epoch):
    #--
    global val
    lr = 3*1e-3

    if val > 160:
        lr = lr/625
    elif val > 120:
        lr = lr/25
    elif val > 60:
        lr  = lr/5
    val+=1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    #--
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=num_classes):
    #--
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    ##

    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    ##

    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  ##

                strides = 2  ##

            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  ##

                ##

                ##

                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    ##

    ##

    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    ##

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    #--
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    ##

    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    ##

    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    ##

    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  ##

                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  ##

                    strides = 2    ##


            ##

            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                ##

                ##

                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    ##

    ##

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    ##

    model = Model(inputs=inputs, outputs=outputs)
    return model


##



def add_to_list(model):
    train_scores = model.evaluate(x_train, y_train, verbose=0, batch_size=512)
    test_scores =  model.evaluate(x_test, y_test, verbose=0, batch_size=512)
    train_losses.append(train_scores[0])
    test_losses.append(test_scores[0])
    test_accuracies.append(test_scores[1])
    train_accuracies.append(train_scores[1])
    return test_scores


##



global_test_accs = []
global_train_accs = []
global_train_losses = []
global_test_losses = []


##





for seed in [10,20]:
    val=0
    np.random.seed(seed)

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  metrics=['accuracy'])

    model.summary()


    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []


    ##

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar100_lookahead_'+str(seed)+'.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    ##

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    callbacks = [lr_scheduler]##


    ##

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        ##

        datagen = ImageDataGenerator(
            ##

            featurewise_center=False,
            ##

            samplewise_center=False,
            ##

            featurewise_std_normalization=False,
            ##

            samplewise_std_normalization=False,
            ##

            zca_whitening=False,
            ##

            zca_epsilon=1e-06,
            ##

            rotation_range=0,
            ##

            width_shift_range=0.1,
            ##

            height_shift_range=0.1,
            ##

            shear_range=0.,
            ##

            zoom_range=0.,
            ##

            channel_shift_range=0.,
            ##

            fill_mode='nearest',
            ##

            cval=0.,
            ##

            horizontal_flip=True,
            ##

            vertical_flip=False,
            ##

            rescale=None,
            ##

            preprocessing_function=None,
            ##

            data_format=None,
            ##

            validation_split=0.0)

        ##

        ##

        print ("Data generating")
        datagen.fit(x_train)

        ##


        cache  = [layer.get_weights() for layer in model.layers]

        for i in range(1,epochs+1):
            a = time.time()

            if i%(k)==0:

                new_weights = [layer.get_weights() for layer in model.layers]
                gradient = compute_gradient(cache, new_weights)
                ##

                update_with_gradient(model, cache, gradient, .5)
                cache  = [layer.get_weights() for layer in model.layers]
                lr_schedule(i)

            else:
                print ("Started training "+str(i))
                model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            epochs=1, verbose=0, workers=4,
                            callbacks=callbacks)

            scores = add_to_list(model)

            b = time.time()
            print ("Epoch ",i ," Test Accuracy: ", scores[1]," time taken ",b-a)


    print ("Finished Model")
    ##

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    global_test_accs.append(test_accuracies)
    global_train_accs.append(train_accuracies)
    global_train_losses.append(train_losses)
    global_test_losses.append(test_losses)


##



import json


##



name = "cifar_100_lookahead"


##



my_json_string = json.dumps(global_train_losses)
with open(name+'train_losses.json', 'w') as f:
    json.dump(my_json_string, f)


##



my_json_string = json.dumps(global_test_losses)
with open(name+'test_losses.json', 'w') as f:
    json.dump(my_json_string, f)


##



my_json_string = json.dumps(global_train_accs)
with open(name+'train_accuracies.json', 'w') as f:
    json.dump(my_json_string, f)


##



my_json_string = json.dumps(global_test_accs)
with open(name+'test_accuracies.json', 'w') as f:
    json.dump(my_json_string, f)


##






##





