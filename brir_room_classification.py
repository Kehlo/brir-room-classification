#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 18:19:19 2022

@author: Christian Kehling
"""

import datetime
import os
import numpy as np
import tensorflow as tf
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from helper import printout, get_tc, get_room_data_samples_and_gt, create_brir_model, print_results


if __name__ == "__main__":

    # init
    fs = 48000
    mel_bands = 512
    test_size = 0.2
    num_epochs = 1000
    batch_size = 64
    learning_rate = 0.008
    beta = 0.999
    patience = 100
    res_folder = "output"

    # load data
    tc = get_tc()
    path2data = []
    roomlist = ['_H1539b', '_H1562', '_H2505', '_HL', '_HU103', '_LR001', '_ML2-102']
    path2data = '/usr/scratch4/chke5810/data/brir_example_data/'
    X, Y = get_room_data_samples_and_gt(path2data, roomlist, fs, True, mel_bands)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    # documentation of the current settings
    printout("\n\nSamplerate: " + str(fs))
    printout("Number of Mel Bands: " + str(mel_bands))

    printout("\n\nShape of Training Data (X_train): " + str(X_train.shape))
    printout("Shape of Training Labels (Y_train): " + str(y_train.shape))
    printout("Shape of Test Data (X_test): " + str(X_test.shape))
    printout("Shape of Test Labels (Y_test): " + str(y_test.shape))
    printout("Data Split Ratio: " + str((1 - test_size) * 100) + "% train : " + str(test_size * 100) + "% test")

    # create model
    inputDims = np.shape(X[0])
    brir_model = create_brir_model(inputDims, len(roomlist))

    # train model
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    opt = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta)
    brir_model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.Accuracy()])
    history = brir_model.fit(x=X_train, y=y_train, verbose=1, batch_size=batch_size, validation_split=0.2, epochs=num_epochs, callbacks=[callback])

    # evaluate the trained models
    loss, acc = brir_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1) #classification
    y_predictions = brir_model.predict(X_test, verbose=1)
    if len(y_test.shape) < 2:
        y_predictions = np.max(y_predictions, axis=1)
    # save/write out
    modelname = tc + '_best_model.h5'
    brir_model.save(os.path.join(res_folder, modelname))
    brir_model.summary(print_fn=printout)

    # statistics print out
    report = sm.classification_report(np.argmax(y_test, axis=1), np.argmax(y_predictions, axis=1), target_names=roomlist)
    acc_test = sm.accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predictions, axis=1))
    f1_test = sm.f1_score(np.argmax(y_test, axis=1), np.argmax(y_predictions, axis=1), average='macro')

    printout(report)
    printout("Test accuracy = " + str(acc_test))
    printout("Test F1-Score = " + str(f1_test))
    printout("\n\n" + str(datetime.datetime.now()))
    printout("EOF")

    # plot conf matrix
    print_results(classes=roomlist, y_p=y_predictions, y_t=y_test, history=history, res_folder=res_folder)

