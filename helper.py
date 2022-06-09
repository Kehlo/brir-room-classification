#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 18:19:19 2022

@author: Christian Kehling
"""


import datetime
import os 
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# create time based filename
tc = str(datetime.datetime.now())
tc = tc.replace(":", "-")
tc = tc.replace(" ", "-")
tc = tc.replace(".", "-")
path = os.getcwd()
outfile = os.path.join(path, "output", (tc + ".txt"))


def get_tc():
    return tc


def printout(content):
    """ Custom function to print output to console and log it to a text file """
    print(content)
    out_path = os.path.split(outfile)[0]
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    with open(outfile, 'a') as logfile:
        print(content, file=logfile)


def get_magnitude_spec(x, sample_rate, fftsize, winsize, hopsize, highest_bin, lowest_bin, log_spec, mel_bands):
    """ Returns the absolute, real valued magnitude spectrogram of the input """
    X = np.abs(tf.signal.stft(x, frame_length=winsize, frame_step=hopsize, fft_length=fftsize))
    if highest_bin is None:
        highest_bin = X.shape[1]
    X = X[:, lowest_bin:highest_bin]

    if mel_bands > 0:
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=fftsize,
            n_mels=mel_bands,
            fmin=0,
            fmax=int(sample_rate/2.0),
            htk=False).T
        X = np.dot(X, mel_basis)

    if log_spec is True:
        X = np.log(1+X)

    return X[0]


def get_room_data_samples_and_gt(rootPath, rList, sr, mono=True, specFeat=128):
    '''
    Custom function to collect all files from the given directory, shorten the
    RIR data to the shortest available RIR length and do frequency transform 
    on the data if specified via specFeat variable. 
    '''
    printout(str(datetime.datetime.now()) + ": Starting Dataset creation ...")

    angleList = ['_000', '_005', '_010', '_015', '_020', '_025', '_030', '_035', '_040', '_045', '_050', '_055',
                 '_060', '_065', '_070', '_075', '_080', '_085', '_090', '_095', '_100', '_105', '_110', '_115',
                 '_120', '_125', '_130', '_135', '_140', '_145', '_150', '_155', '_160', '_165', '_170', '_175',
                 '_180', '_185', '_190', '_195', '_200', '_205', '_210', '_215', '_220', '_225', '_230', '_235',
                 '_240', '_245', '_250', '_255', '_260', '_265', '_270', '_275', '_280', '_285', '_290', '_295',
                 '_300', '_305', '_310', '_315', '_320', '_325', '_330', '_335', '_340', '_345', '_350', '_355']
    suffix = '.wav'

    locList = []
    gtList1hot = []
    gtList = []
    for room in rList:
        # create GT vector
        onehot = np.zeros((1, len(rList)), dtype=np.int8)
        hotPos = rList.index(room)
        onehot[0][hotPos] = 1
        gtVec = onehot[0]
        for angle in angleList:
            filename = 'BRIR_DH' + room + '_M_C' + angle + suffix
            location = os.path.join(rootPath, filename)
            locList.append(location)
            gtList.append(hotPos)
            gtList1hot.append(gtVec)

    # load audio
    printout(str(datetime.datetime.now()) + ": Loading BRIR files from disk ...")
    X = []
    minLength = 480000
    for locname in locList:
        x, sr = librosa.load(locname, mono=False, sr=sr)
        if mono:
            x = x[0]
        x = x/np.max(np.abs(x))
        X.append(x)
        xlength = len(x)
        if xlength < minLength:
            minLength = xlength

    # correct length to shortest rir length and tf transform if needed
    for n in range(len(X)):
        if specFeat > 0:
            X[n] = get_magnitude_spec(X[n][0:minLength], sr, minLength, minLength, minLength, None, None, True, specFeat)
        else:
            X[n] = X[n][0:minLength]

    # check ground truth vector for keras post processing
    if len(gtVec) < 3:
        y_out = gtList
    else:
        y_out = gtList1hot

    printout(str(datetime.datetime.now()) + ": Dataset Extraction finished.")
    return [np.asarray(X), np.asarray(y_out)]


def  create_brir_model(inputShape, outputShape):
    """ Create a simple DNN network """
    model = Sequential()
    model.add(Dense(32, input_shape=inputShape, activation='relu'))
    model.add(Dense(32,  activation='relu'))
    model.add(Dense(outputShape, activation='softmax'))
    return model


def print_results(classes, y_p, y_t, history, res_folder):
    """ Custom print function for brir_room_classificaton.py"""
    if len(y_t.shape) > 1:
        cm = sm.confusion_matrix(np.argmax(y_t, axis=1), np.argmax(y_p, axis=1))
    else:
        cm = sm.confusion_matrix(y_t, y_p)
    cmsum = 1/cm.sum(axis=1)
    cm_norm = np.multiply(cm, cmsum[:, None])
    
    printout(cm_norm)
    
    fig, ax = plt.subplots(figsize=(10,10))
    cmd = sm.ConfusionMatrixDisplay(cm_norm, display_labels=classes)
    cmd.plot(cmap='binary', values_format='.2%', ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(res_folder, 'confusion_matrix_'+str(tc)+'.png'))
    
    # plot history
    plt.clf() 
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(res_folder, 'acc_'+str(tc)+'.png'))
    
    plt.clf() 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(res_folder, 'loss_'+str(tc)+'.png'))


    # plot roc curve
    if len(y_t.shape) < 2:
        plt.clf()
        fpr, tpr, thresholds = sm.roc_curve(y_t, y_p)
        roc_auc = sm.auc(fpr, tpr)
        display = sm.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
        display.plot()
        plt.savefig(os.path.join(res_folder, 'ROC_curve_' + str(tc) + '.png'))