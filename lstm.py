#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 07:46:42 2018

@author: Thomas
"""

"""
Prepare MIDI file and feed it to neural network for training
"""


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

"""
A Recurrent Neural Network is a type of artificial neural network that makes
use of sequential data. In this model we will use Long Short Term Memory (LSTM) 
network. They are useful when the network needs to remember information for a 
long period of time as in case of music and text generation.
"""

def create_network(network_input, n_vocab):
    # Create the structure of LSTM
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def prepare_sequences(notes, n_vocab, temp, n_vocab2, sequence_length):
    """ Prepare data to be used by the Neural Network """

    # get all pitch names
    pitchnames1 = sorted(set(item for item in notes))
    pitchnames2 = sorted(set(item for item in temp))
    
     # create a dictionary to map pitches to integers
    note_to_int1 = dict((note, number) for number, note in enumerate(pitchnames1))
    note_to_int2 = dict((temp, number) for number, temp in enumerate(pitchnames2))
    network_input = []
    network_output_note = []
    network_output_time = []
    seqInAll=[]
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        sequence_in2 = temp[i:i + sequence_length]
        sequence_out2 = temp[i + sequence_length]
        seq1=([note_to_int1[char] for char in sequence_in])
        seq2=([note_to_int2[char] for char in sequence_in2])
        network_output_note.append(note_to_int1[sequence_out])
        network_output_time.append(note_to_int2[sequence_out2])
        seqIn=np.transpose([seq1,seq2])
        seqInAll.append(seqIn)

    
    
    n_patterns = len(seqInAll)
    seqInAll=np.array(seqInAll)
    
    network_input = np.reshape(seqInAll, (n_patterns, sequence_length, 2))
    
    # normalize input
    network_input = network_input / float(max(n_vocab,n_vocab2))
    network_output_note = np_utils.to_categorical(network_output_note)
    network_output_time = np_utils.to_categorical(network_output_time)

    return (network_input, network_output_note,network_output_time,pitchnames1,pitchnames2)


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    #model.fit(network_input, network_output, callbacks=callbacks_list)
    model.fit(network_input, network_output, epochs=70, batch_size=32, callbacks=callbacks_list)

def predict(data_input, model_note, model_time, pitchnames1, pitchnames2):
    """Predict notes """
    n_vocab1=len(pitchnames1)
    n_vocab2=len(pitchnames2)
    
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames1))
    int_to_time = dict((number, note) for number, note in enumerate(pitchnames2))
    
    
    start = np.random.randint(0, len(data_input)-1)
    
    pattern = data_input[start]
    prediction_note = []
    prediction_time = []
    
    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 2))
        #prediction_input = prediction_input / float(max(n_vocab1,n_vocab2))
        
        predict_note = model_note.predict(prediction_input, verbose=0)
        predict_time = model_time.predict(prediction_input, verbose=0)
    
        index_note = np.argmax(predict_note)
        index_time = np.argmax(predict_time)
        
        result_note = int_to_note[index_note]
        result_time = int_to_time[index_time]
        
        prediction_note.append(result_note)
        prediction_time.append(result_time)
        append_line=np.reshape([index_note,index_time],(1,2))/ float(max(n_vocab1,n_vocab2))
        pattern=np.append(pattern,append_line,axis=0)
        pattern = pattern[1:len(pattern)]
        return(prediction_note , prediction_time)
    
    
def export_song(prediction_note , prediction_time):
    """export data to a midi format"""
    import pandas 
    from music21 import instrument, note, stream, chord
    
    data = pandas.read_csv('CaseToNote.csv',sep=',')
    ConvertNote=data['Note']
    offset = 0
    output_song = []
    
    # create note and chord objects based on the values generated by the model
    for iTime in range(len(prediction_note)):
        groupNote=prediction_note[iTime]
        time=prediction_time[iTime]
        # split groupNote
        notesInGroup = groupNote.split('.')
        notesInGroup = notesInGroup[-len(notesInGroup)+1:]
        notes = []
        for current_note in notesInGroup:
            noteConv=ConvertNote[min(int(current_note),44)] #Max string note in the set
            new_note = note.Note(noteConv)
            new_note.storedInstrument = instrument.ElectricGuitar()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_song.append(new_chord)
    
            # increase offset each iteration so that notes do not stack
        offset += time/2
    
    midi_stream = stream.Stream(output_song)
    #midi_stream=instrument.ElectricGuitar(midi_stream)
    midi_stream.write('midi', fp='test_output.mid')