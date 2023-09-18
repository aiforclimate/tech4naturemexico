"""
Tech4Nature Project

This python file contains the neural networks architectures used
as an embedding network, they take an image as input
and produce a vector as outputç

we are going to try different kinds of Neural Network Architectures 

Lets Explore different architectures 

"""


import tensorflow as tf

import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda,UpSampling2D,Conv2D,MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16

imsize = 224


def initialize_base_branch_512vector():
    #this is the model used to make the embeddings
    #the input is 128x128x3
    input = Input(shape=(imsize,imsize,3), name="base_input")
    #convolutional feature extractor

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(input)
    x = layers.AveragePooling2D()(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)


    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    #change to 512 
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)

    #Dense Layer
    x = layers.Flatten(name="flatten_input")(x)
    #x = layers.Dense(8192, name="first_base_dense")(x)
    x = layers.Dense(1024,activation='relu', name="first_base_dense")(x)
    x = layers.Dropout(0.5)(x)  # Adding a dropout layer with a dropout rate of 0.5
    #x = layers.Dense(4096, name="second_base_dense")(x)
    x = layers.Dense(1024,activation='relu', name="second_base_dense")(x)
    x = layers.Dropout(0.5)(x)  # Adding another dropout layer
    #x = layers.Dense(2048, activation='relu', name="third_base_dense")(x)
    x = layers.Dense(512, name="third_base_dense")(x)

    #Returning a Model, with input and outputs, not just a group of layers.
    return Model(inputs=input, outputs=x)






def initialize_base_branch_1024vector():
    #this is the model used to make the embeddings
    #the input is 128x128x3
    input = Input(shape=(imsize,imsize,3), name="base_input")
    #convolutional feature extractor

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(input)
    x = layers.AveragePooling2D()(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)


    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    #change to 512 
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)

    #Dense Layer
    x = layers.Flatten(name="flatten_input")(x)
    #x = layers.Dense(8192, name="first_base_dense")(x)
    x = layers.Dense(2048,activation='relu', name="first_base_dense")(x)
    x = layers.Dropout(0.5)(x)  # Adding a dropout layer with a dropout rate of 0.5
    #x = layers.Dense(4096, name="second_base_dense")(x)
    x = layers.Dense(2048,activation='relu', name="second_base_dense")(x)
    x = layers.Dropout(0.5)(x)  # Adding another dropout layer
    #x = layers.Dense(2048, activation='relu', name="third_base_dense")(x)
    x = layers.Dense(1024, name="third_base_dense")(x)

    #Returning a Model, with input and outputs, not just a group of layers.
    return Model(inputs=input, outputs=x)




def initialize_base_branch_1024_v2_vector():
    #this is the model used to make the embeddings
    #the input is 128x128x3
    input = Input(shape=(imsize,imsize,3), name="base_input")
    #convolutional feature extractor

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(input)
    x = layers.AveragePooling2D()(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)


    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    #change to 512 
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)

    #Dense Layer
    x = layers.Flatten(name="flatten_input")(x)
    #x = layers.Dense(8192, name="first_base_dense")(x)
    x = layers.Dense(2048,activation='relu', name="first_base_dense")(x)
    x = layers.Dropout(0.5)(x)  # Adding a dropout layer with a dropout rate of 0.5
    #x = layers.Dense(4096, name="second_base_dense")(x)
    x = layers.Dense(1024,activation='relu', name="second_base_dense")(x)
    x = layers.Dropout(0.5)(x)  # Adding another dropout layer
    #x = layers.Dense(2048, activation='relu', name="third_base_dense")(x)
    x = layers.Dense(1024, name="third_base_dense")(x)

    #Returning a Model, with input and outputs, not just a group of layers.
    return Model(inputs=input, outputs=x)






def initialize_base_branch_1024_v2_vector_load():
    #this is the model used to make the embeddings
    #the input is 128x128x3
    input = Input(shape=(imsize,imsize,3), name="base_input")
    #convolutional feature extractor

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(input)
    x = layers.AveragePooling2D()(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)


    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    #change to 512 
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)

    #Dense Layer
    x = layers.Flatten(name="flatten_input")(x)
    #x = layers.Dense(8192, name="first_base_dense")(x)
    x = layers.Dense(2048,activation='relu', name="first_base_dense")(x)
    x = layers.Dropout(0.5)(x)  # Adding a dropout layer with a dropout rate of 0.5
    #x = layers.Dense(4096, name="second_base_dense")(x)
    x = layers.Dense(1024,activation='relu', name="second_base_dense")(x)
    x = layers.Dropout(0.5)(x)  # Adding another dropout layer
    #x = layers.Dense(2048, activation='relu', name="third_base_dense")(x)
    x = layers.Dense(1024, name="third_base_dense")(x)

    #Returning a Model, with input and outputs, not just a group of layers.
    return Model(inputs=input, outputs=x)




def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def triplet_loss(x, alpha = 0.2):
    # Triplet Loss function.
    anchor,positive,negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss




def get_convolutional_layers(model):
    convolutional_layers = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            convolutional_layers.append(layer)
        elif isinstance(layer, Model):  # Handle functional sub-models
            convolutional_layers.extend(get_convolutional_layers(layer))

    return convolutional_layers




# Model structure
def initialize_base_branch_vgg16():
    #this is the model used to make the embeddings
    #the input is 128x128x3
    input = Input(shape=(imsize,imsize,3), name="base_input")
    
    vgg = VGG16(weights="imagenet",
            include_top=False,
            input_tensor=input)
    vgg.trainable = False
    # flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)
    
    embedding = Dense(2000,activation="relu")(flatten)
    embedding = layers.Dropout(0.3, name="f_dropout")(embedding)
    embedding = Dense(1500,activation="relu")(embedding)
    embedding = layers.Dropout(0.3, name="s_dropout")(embedding)
    embedding = Dense(1000,activation="relu")(embedding)
    embedding = layers.Dropout(0.3, name="t_dropout")(embedding)
    embedding = Dense(1000, name="IMGembedding")(embedding)


    #convolutional feature extractor
    # input
    #Returning a Model, with input and outputs, not just a group of layers.
    return Model(inputs=input, outputs=embedding)





# Model structure
def initialize_base_branch_vgg16_v2_1024():
    #this is the model used to make the embeddings
    #the input is 128x128x3
    input = Input(shape=(imsize,imsize,3), name="base_input")
    
    vgg = VGG16(weights="imagenet",
            include_top=False,
            input_tensor=input)
    vgg.trainable = False
    # flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)
    
    #embedding = Dense(2000,activation="relu")(flatten)
    #embedding = layers.Dropout(0.3, name="f_dropout")(embedding)
    embedding = Dense(2048,activation="relu")(flatten)
    embedding = layers.Dropout(0.3, name="s_dropout")(embedding)
    embedding = Dense(2048,activation="relu")(embedding)
    embedding = layers.Dropout(0.3, name="t_dropout")(embedding)
    embedding = Dense(1024, name="IMGembedding")(embedding)


    #convolutional feature extractor
    # input
    #Returning a Model, with input and outputs, not just a group of layers.
    return Model(inputs=input, outputs=embedding)




def initialize_base_branch_vgg16_latent_space():
    TRAINABLE=False
    input = Input(shape=(imsize,imsize,3), name="base_input")
    base_model = VGG16(weights='imagenet',include_top=False,input_tensor=input)
    

    for layer in base_model.layers:
        layer.trainable=TRAINABLE
    
#-------------------encoder---------------------------- 
#--------(pretrained & trainable if selected)----------

#    block1
    x=base_model.get_layer('block1_conv1')(input)
    x=base_model.get_layer('block1_conv2')(x)
    x=base_model.get_layer('block1_pool')(x)

#    block2
    x=base_model.get_layer('block2_conv1')(x)
    x=base_model.get_layer('block2_conv2')(x)
    x=base_model.get_layer('block2_pool')(x)

#    block3
    x=base_model.get_layer('block3_conv1')(x)
    x=base_model.get_layer('block3_conv2')(x)
    x=base_model.get_layer('block3_conv3')(x)    
    x=base_model.get_layer('block3_pool')(x)

#    block4
    x=base_model.get_layer('block4_conv1')(x)
    x=base_model.get_layer('block4_conv2')(x)
    x=base_model.get_layer('block4_conv3')(x)    
    x=base_model.get_layer('block4_pool')(x)

#    block5
    x=base_model.get_layer('block5_conv1')(x)
    x=base_model.get_layer('block5_conv2')(x)
    x=base_model.get_layer('block5_conv3')(x)
     
    
#--------latent space (trainable) ------------
    x=base_model.get_layer('block5_pool')(x)     
    x = Conv2D(512, (3, 3), activation='relu', padding='same',name='latent')(x)
    x = UpSampling2D((2,2))(x)  
    
#--------------Dense Layer (trainable)----------- 
    flatten = Flatten()(x)

    
    embedding = Dense(2048,activation="relu")(flatten)
    embedding = layers.Dropout(0.3, name="s_dropout")(embedding)
    embedding = Dense(2048,activation="relu")(embedding)
    embedding = layers.Dropout(0.3, name="t_dropout")(embedding)
    embedding = Dense(1024, name="IMGembedding")(embedding)    

    #return 
    return Model(inputs=input, outputs=embedding)