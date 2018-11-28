#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:09:25 2018

@author: Thomas
"""

import lstm
import pickle


"""Part 1, prepare & train models"""
#Import Data
with open('data/tab', 'rb') as filepath:
        notes = pickle.load(filepath)
        
with open('data/temp', 'rb') as filepath:
        temp = pickle.load(filepath)

# get amount of pitch names
n_vocab1 = len(set(notes))
n_vocab2 = len(set(temp))
seq=12
#Prepare data
(network_input, network_output_note,network_output_time,pitchnames1,pitchnames2) = lstm.prepare_sequences(notes, n_vocab1, temp, n_vocab2,seq)

#Create models
model_note = lstm.create_network(network_input,n_vocab1)
model_time = lstm.create_network(network_input,n_vocab2)

#Train models
lstm.train(model_note, network_input, network_output_note)
lstm.train(model_time, network_input, network_output_time)


""" ¨Part 2 Predict """

#Load weights created after training
model_note.load_weights('weigths_note.hdf5')
model_time.load_weights('weigths_time.hdf5')

#Predict note using random part of input & both models
(prediction_note , prediction_time) = lstm.predict(network_input, model_note, model_time, pitchnames1, pitchnames2)


#Export song on a mdi format
lstm.export_song(prediction_note , prediction_time)
