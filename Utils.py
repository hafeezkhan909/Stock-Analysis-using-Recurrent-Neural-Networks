import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers


def MinMaxScaler(data):    

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-8)
    return norm_data


def performance(test_y, test_y_hat, metric_name):

    assert metric_name in ['mse', 'mae']
  
    if metric_name == 'mse':
        score = mean_squared_error(test_y, test_y_hat)
    elif metric_name == 'mae':
        score = mean_absolute_error(test_y, test_y_hat)
        
    score = np.round(score, 4)
    
    return score


def binary_cross_entropy_loss (y_true, y_pred):

  # Exclude masked labels
    idx = tf.cast((y_true >= 0), float)
  # Cross entropy loss excluding masked labels
    loss = -(idx * y_true * tf.math.log(y_pred) + idx * (1-y_true) * tf.math.log(1-y_pred))
    return loss


def mse_loss (y_true, y_pred):

 
    idx = tf.cast((y_true >= 0), float)
 
    loss = idx * tf.pow(y_true - y_pred, 2)
    return loss


def rnn_sequential (model, model_name, h_dim, return_seq):

  
    if model_name == 'rnn':
        model.add(layers.SimpleRNN(h_dim, return_sequences=return_seq))
    elif model_name == 'lstm':
        model.add(layers.LSTM(h_dim, return_sequences=return_seq))
    elif model_name == 'gru':
        model.add(layers.GRU(h_dim, return_sequences=return_seq))
    
    return model

