#! -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:36:49 2016

@author: mangotee
"""

import os
import sys
import fnmatch
import socket
import time

import numpy as np
import pandas as pd
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn as skl

from scipy import signal

from wristbandDataReader import * 


def plot_confusion_matrix(cm, title='Confusion matrix', target_names=None, support=None, cmap=plt.cm.Blues, scores=None):
    #hFig = mpl.figure()   
    if support is not None:
        cm = 100*(np.float32(cm.T)/support).T
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if scores is not None:
        strscores = 'Prec: %0.3f, Rec: %0.3f, fMeas: %0.3f' % (scores[0],scores[1],scores[2])  
        title = title + ' \n ' + strscores
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')# debug feature extraction

class numpyDataProvider(object):
    def __init__(self,pandata=None,params=None):
        self.setParams()
        if pandata is not None:
            self.dataFromPandata(pandata)
        if params is not None:
            self.setParams(nrClasses=params['nrClasses'])
    
    def getEmptyParamsStruct(self):
        params = {}
        params['nrClasses'] = 10
        return params
        
    def setParams(self,nrClasses=10):
        self.nrClasses = nrClasses
        
    def setParamsFromStruct(self,params):
        self.nrClasses = params['nrClasses']
    
    def setDataFromPandata(self,pandata,dataCols,labelCols,normalize=True):
        nrData = len(pandata)
        self.data = []
        self.labels = []
        for i in range(nrData):
            self.data.append(np.squeeze(pandata[i].ix[:,dataCols].values))
            self.labels.append(np.squeeze(pandata[i].ix[:,labelCols].values))
        self.featDim = self.data[0].shape[1]
        if len(self.labels[0].shape)==1:
            self.labelDim = 1
        else:                
            self.labelDim = self.labels[0].shape[1]
        if normalize:
            self.normalizeData_ZeroMeanUnitVariance()
            #self.normalizeData_DivideByConstantScalar()
        
    def setDataFromNumpyData(self,data,labels,normalize=True):
        self.data = data
        self.labels = labels
        self.featDim = self.data[0].shape[1]
        if len(self.labels[0].shape)==1:
            self.labelDim = 1
        else:                
            self.labelDim = self.labels[0].shape[1]
        if normalize:
            self.normalizeData_ZeroMeanUnitVariance()
            #self.normalizeData_DivideByConstantScalar()

    def dataSwitchToSpectroGrams(self, fs=62.0, nperseg=256, noverlap=128, maxFreqToConsider=10, axis=0):        
        dataList = []
        labelList = [] 
        spectralTimes = [] 
        spectralFreqs = []
        for i in range(len(self.data)):        
            x = self.data[i][:,0]
            f, t, Sxx = signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=True, axis=axis, scaling='density', mode='psd')        
            Sxx = Sxx.T
            maxFreqIdx = np.argmax(f>maxFreqToConsider)
            data = Sxx[:,0:maxFreqIdx]
            for col in range(1,self.data[i].shape[1]):       
                x = self.data[i][:,col]
                f, t, Sxx = signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=True, axis=axis, scaling='density', mode='psd')
                Sxx = Sxx.T
                data = np.hstack((data,Sxx[:,0:maxFreqIdx]))    
            dataList.append(data)
            labels = self.labels[i][np.int64(t*fs)]
            labelList.append(labels)
            spectralTimes.append(t)
            spectralFreqs.append(f)
        self.featDim = dataList[0].shape[1]
        self.data = dataList
        self.labels = labelList
        self.spectralTimes = spectralTimes
        self.spectralFreqs = spectralFreqs
        self.normalizeData_ZeroMeanUnitVariance()
        #self.normalizeData_DivideByConstantScalar(s=1.0)
        
    def timeIdxFromSpectralIdx(self,dataIdx,spectralIdx):
        return self.spectralTimes[dataIdx][spectralIdx] 
                
    def normalizeData_ZeroMeanUnitVariance(self):
        for i in range(len(self.data)):
            D = self.data[i]            
            # de-mean
            D = D - np.mean(D,axis=0)
            # unit standard deviation
            D = D / np.std(D,axis=0)
            self.data[i] = D
            
    def normalizeData_DivideByConstantScalar(self,s=100.0):
        for i in range(len(self.data)):
            D = self.data[i]            
            # de-mean
            D = D / s
            self.data[i] = D


#%
# read in all data at once and store in list "pandata"
# parameters (machine-dependent) for allocating a list of expert and wristband files 
dataroot = 'path/to/data/root/folder'
dataroot = '/Users/mangotee/PostDoc/Projects/PDwristband/data/long'

pattern = 'PT17.00*'  # only Parkinson patients, not healthy controls (those would be 'PT17.1*')
dataFileTable = collectDataAndExpertFiles(dataroot, pattern=pattern)

# data settings
params = setDataParameters(['accelerometerN', 'gyroscopeN'],# data columns, here: L2-norm or accel. and gyr., can be any combination of wristband csv data columns
                               ['4classPD3'])                   # label columns (i.e. target(s) for Keras algorithm, e.g. CNN/RNN), can be any combination of excel file columns

# read in data
pandata = []

idxProblemsLoad = []
idxProblemsMerge = []

print 'Reading training data:'
trainingDataIdx = range(20) 
npdatatrain = []
nplabelstrain = []
for idx in trainingDataIdx: 
    # load data into pandas DataFrames
    try:    
        dataBand, dataExperts = readSubjectDataAll(dataFileTable['ExpertFiles'][idx], dataFileTable['WristbandFiles'][idx])
    except: 
        print 'Load problems at idx=%d' % idx        
        idxProblemsLoad.append(idx)
        continue
    # use pandas merge tools to merge wristband data and expert labels 
    try: 
        dataMerged = synchronizeDataAndLabels(dataBand, dataExperts)
    except: 
        print 'Merge problems at idx=%d' % idx
        idxProblemsMerge.append(idx)
        continue
    # load data into pandas DataFrames
    npdatatrain.append(np.squeeze(dataMerged.ix[:,params['dataCols']].values))
    nplabelstrain.append(np.squeeze(dataMerged.ix[:,params['labelCols']].values))
    print 'Merge of %d successful.' % idx

print 'Reading testing data:'
testingDataIdx = range(20,27)
npdatatest = []
nplabelstest = []
for idx in testingDataIdx: 
    # load data into pandas DataFrames
    try:   
        dataBand, dataExperts = readSubjectDataAll(dataFileTable['ExpertFiles'][idx], dataFileTable['WristbandFiles'][idx])
    except: 
        print 'Load problems at idx=%d' % idx        
        idxProblemsLoad.append(idx)
        continue
    # use pandas merge tools to merge wristband data and expert labels         
    try:
        dataMerged = synchronizeDataAndLabels(dataBand, dataExperts)
    except: 
        print 'Merge problems at idx=%d' % idx
        idxProblemsMerge.append(idx)
        continue
    npdatatest.append(np.squeeze(dataMerged.ix[:,params['dataCols']].values))
    nplabelstest.append(np.squeeze(dataMerged.ix[:,params['labelCols']].values))
    print 'Merge of %d successful.' % idx

#%
# feed a data provider object with data and (optionally) convert raw data to histograms
print 'Switch training data to spectrograms.'
objDTrain = numpyDataProvider()
dataParams = objDTrain.getEmptyParamsStruct()
dataParams['nrClasses'] = 4
objDTrain.setParamsFromStruct(dataParams)
objDTrain.setDataFromNumpyData(npdatatrain,nplabelstrain)
objDTrain.dataSwitchToSpectroGrams(fs=62.0, nperseg=256, noverlap=128, maxFreqToConsider=10.0)

# same for test data
print 'Switch test data to spectrograms.'
objDTest = numpyDataProvider()
dataParams = objDTest.getEmptyParamsStruct()
dataParams['nrClasses'] = 4
objDTest.setParamsFromStruct(dataParams)
objDTest.setDataFromNumpyData(npdatatest,nplabelstest)
objDTest.dataSwitchToSpectroGrams(fs=62.0, nperseg=256, noverlap=128, maxFreqToConsider=10.0)

print 'All done.'

print '\n\n\n'

print 'Data is now available in data providers objDTrain and objDTest.' 
print 'You can access data via objD.data, e.g. objDTrain.data[0].shape yields:'
print objDTrain.data[0].shape
print 'You can access labels (i.e. learning targets) via objD.labels, e.g. objDTrain.labels[0].shape yields: ...'
print objDTrain.labels[0].shape
print '... and np.unique(objDTrain.labels[0]) yields:'
print np.unique(objDTrain.labels[0])
print "Please note: '?' encodes a missing-value flag (happened if human expert rater e.g. went to lunch break)"