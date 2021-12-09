import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from tools import pretraitement
import random
import os
import json
import sys
import time

# Model manipulation
## Construction d'un CNN
def create_model(dropout=0.5):
    input_shape=(32, 32, 3)

    # La base CNN
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2))) # reduce to 16*16*3

    model.add(layers.Conv2D(64, (3, 3),padding='same', activation='relu'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(64, (3, 3),padding='same', activation='relu'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2))) # reduce to 8*8*3

    model.add(layers.Conv2D(128, (3, 3),padding='same', activation='relu'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(128, (3, 3),padding='same', activation='relu'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))  # reduce to 4*4*3

    # Ajout de couches denses vers la fin du model
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout/2))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

def create_datagen():
    param = {'rotation_range' : 15,
                'horizontal_flip' : True,
                'vertical_flip' : False,
                'width_shift_range' : 0.3,
                'height_shift_range' : 0.3
    }

    datagen = ImageDataGenerator(**param)
    return datagen, param

# Compiling model
def compil_model(model):
    ## training parameters
    learning_rate = 0.001
    loss='mean_squared_error'
    #loss='categorical_crossentropy'
    optimizers={'adam':Adam(learning_rate=learning_rate)}
    optimizer = 'adam'
    metrics=['accuracy']
    epochs=40
    batch_size=128
    
    model.compile(optimizer=optimizers[optimizer],
                  loss=loss,
                  metrics=metrics)
    
    return model, {
        'learning_rate': learning_rate,
        'loss': loss,
        'optimizer': optimizer,
        'metrics': metrics,
        'epochs': epochs,
        'batch_size': batch_size
    }

# Plotting
def plot_accuracy(history,h2=None):
    if (h2):
        plt.plot(history.history['accuracy']+h2.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy']+h2.history['val_accuracy'], label = 'val_accuracy')
    else:
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
def plot_loss(history,h2=None):
    if (h2):
        plt.plot(history.history['loss']+h2.history['loss'], label='loss')
        plt.plot(history.history['val_loss']+h2.history['val_loss'], label = 'val_loss')
    else:
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

# json file containing all model informations
def save_in_json(file_name, model, parametres):
    d = '/'.join(file_name.split('/')[0:-1])
    if d:
        d += '/'
        
    if (not os.path.isfile(file_name)):
        data = {}
    else:
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)
    t=str(time.time()).split('.')[0]
    name = d+f'model-{t}.h5'
    
    data[f'model-{t}.h5'] = {
        'time': str(time.time()).split('.')[0],
        'structure': parametres['structure'],
        'training': parametres['training'],
        'data': parametres['data'],
        'results': parametres['results']
    }
    
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file)
    
    # save the model in the file
    model.save(name)

# Parametres informations

def get_structure(model):
    model_config = json.loads(model.to_json())

    layers_information = []
    l=model_config['config']['layers']

    for i in range(len(l)):
        layers_information.append({
            'type': l[i]['class_name'],
            'config': l[i]['config']
        })

    return {'layers' : layers_information}

