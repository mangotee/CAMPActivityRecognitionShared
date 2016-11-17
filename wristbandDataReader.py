#! -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:36:49 2016

@author: mangotee
"""

import numpy as np
import pandas as pd
import warnings

import os
import sys
import fnmatch
import socket


def locateFiles(pattern, root_path):
    fl = np.array([])
    for root, dirs, files in os.walk(root_path):
        for filename in fnmatch.filter(files, pattern):
            #print( os.path.join(root, filename))
            fl = np.append(fl,os.path.join(root, filename))
    return fl


def locateDirs(pattern, root_path, level=0):
    pl = np.array([])
    for root, dirs, files in walklevel(root_path, level): #os.walk(root_path):
        for pathname in fnmatch.filter(dirs, pattern):
            #print( os.path.join(root, pathname))
            pl = np.append(pl,os.path.join(root, pathname))
    return pl


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def collectDataAndExpertFiles(dataroot, pattern='*'):
    # instantiate an empty dataframe     
    columnNames = ['SubjectID','Dir','ExpertFiles','WristbandFiles']
    dataFileTable = pd.DataFrame(columns=columnNames)    
    # get the pathlist for all subjects
    subjectPaths = locateDirs(pattern, dataroot, level=0)
    dataFileTable['Dir'] = subjectPaths
    # Fill remaining dataframe column
    for idx, pn in enumerate(subjectPaths):
        # 'SubjectID' column
        dump, subjectID = os.path.split(pn)
        dataFileTable['SubjectID'][idx] = subjectID
        # 'ExpertFiles' column   
        dataFileTable['ExpertFiles'][idx] = locateFiles(subjectID+'*.xlsx', pn)
        # 'WristbandFiles'column
        dataFileTable['WristbandFiles'][idx] = locateFiles(subjectID+'*.csv', pn)
    return dataFileTable        


# AHMAD
def filelistWristband(filename):
    # returns a pandas dataframe
    df = pd.read_csv(filename, delimiter=',', dtype=None, header=0)
    # drop all rows with a non-valid timestamp    
    df = df.dropna(subset=['time'])
    # accelerometer/gyroscope magnitudes
    # df.loc[rows,cols] is the syntax for slicing of pandas dataframes
    df['accelerometerN'] = np.linalg.norm(df.loc[:,['accelerometerX','accelerometerY','accelerometerZ']].values, axis=1)
    df['gyroscopeN'] = np.linalg.norm(df.loc[:,['gyroscopeX','gyroscopeY','gyroscopeZ']].values, axis=1)
    # re-label timestamps to a regular sampling between first and last timestamp in the recording 
    # done via linspace, i.e. the new frequency corresponds to the average timestamp step
    time_regular = np.round(np.linspace(df.iloc[0]['time'], df.iloc[-1]['time'], num=df.shape[0]))
    # create an index for the dataframe from the regular timestamps (i.e. now, there are no duplicate timestamps anymore)
    df['time_regular'] = time_regular
    df['timeIx'] = pd.to_datetime(df['time_regular'],unit='ms') #.tz_localize('dateutil/Etc/GMT').tz_convert('dateutil/Europe/Berlin')
    df = df.set_index('timeIx')
    return df


# AHMAD
def readWristbandCSV(filename):
    # returns a pandas dataframe
    df = pd.read_csv(filename, delimiter=',', dtype=None, header=0)
    # drop all rows with a non-valid timestamp    
    df = df.dropna(subset=['time'])
    # accelerometer/gyroscope magnitudes
    # df.loc[rows,cols] is the syntax for slicing of pandas dataframes
    df['accelerometerN'] = np.linalg.norm(df.loc[:,['accelerometerX','accelerometerY','accelerometerZ']].values, axis=1)
    df['gyroscopeN'] = np.linalg.norm(df.loc[:,['gyroscopeX','gyroscopeY','gyroscopeZ']].values, axis=1)
    # re-label timestamps to a regular sampling between first and last timestamp in the recording 
    # done via linspace, i.e. the new frequency corresponds to the average timestamp step
    time_regular = np.round(np.linspace(df.iloc[0]['time'], df.iloc[-1]['time'], num=df.shape[0]))
    # create an index for the dataframe from the regular timestamps (i.e. now, there are no duplicate timestamps anymore)
    df['time_regular'] = time_regular
    df['timeIx'] = pd.to_datetime(df['time_regular'],unit='ms') #.tz_localize('dateutil/Etc/GMT').tz_convert('dateutil/Europe/Berlin')
    df = df.set_index('timeIx')
    return df


# AHMAD
def readExpertsExcel(filename):
    df = pd.read_excel(filename, sheetname=1)
    df['symptoms'] = df['symptoms'].fillna(value='none')
    df['PD medication'] = df['PD medication'].fillna(value='none')
    df['other medication'] = df['other medication'].fillna(value='none')
    # we only take rows in which the UNIX time and PD3 are not NaN
    df = df.dropna(subset=['UNIX time'])
    df = df.dropna(subset=['PD3'])
    df['UNIX time'] = df['UNIX time'].values * 1000.0
    df['timeIx'] = pd.to_datetime(df['UNIX time'],unit='ms')
    df = df.set_index('timeIx')
    return df


def readSubjectDataAll(filelistExpert,filelistWristband):   
    # experts files    
    dflist = []    
    for fn in filelistExpert:
        df = readExpertsExcel(fn)
        df = createClassesFromLongData_4classPD3(df)
        dflist.append(df)
    dfExperts = pd.concat(dflist)
    dfExperts = dfExperts.reset_index().drop_duplicates(subset='timeIx', keep='last').set_index('timeIx')
    dfExperts.sort_index(inplace=True)
    # wristband files    
    dflist = []    
    for fn in filelistWristband:
        df = readWristbandCSV(fn)
        dflist.append(df)
    dfWristband = pd.concat(dflist)
    dfWristband = dfWristband.reset_index().drop_duplicates(subset='timeIx', keep='last').set_index('timeIx')
    dfWristband.sort_index(inplace=True)
    # merge the two
    #dfMerged = synchronizeDataAndLabels(dfWristband,dfExperts)
    return dfWristband, dfExperts
    

# FELIX
def classFromLabelName(somelabel):
    # Felix
    # classes
    # regressionNumbers
    classIDs =\
        {
            'dyskinesia': 1,
            'bradykinesia': 2,
            'tremor': 3,
            'balanced': 4,
            '?': None
        }
    return classIDs[somelabel]

def classFromLabelNameThreeClass(somelabel):
    # Felix
    # classes
    # regressionNumbers
    classIDs =\
        {
            'dyskinesia': 1,
            'bradykinesia': 2,
            'tremor': 2,
            'balanced': 3,
            '?': None
        }
    return classIDs[somelabel]


# AHMAD
def synchronizeDataAndLabels(dfWristband,dfExperts):
    # both need a column called UnixTime
    df = pd.concat([dfWristband, dfExperts], axis=1)
    # forward-fill all columns from dfExperts into the higher frequency
    df.loc[:,dfExperts.columns.values.tolist()] = df.loc[:,dfExperts.columns.values.tolist()].fillna(method='ffill')
    # get rid of all the rows in which there is no "signal" (i.e. all rows with second-exact timestamps as in df1)
    df = df.dropna(subset=['accelerometerX'])
    # now get rid of all the rows in which signal data is still unlabeled 
    df = df.dropna(subset=['PD3'])
    return df


# AHMAD
def createClassesFromLongData_4classPD3(df):
    df['4classPD3'] = ''
    
    # DYSKINESIA    
    # Generally, PD3>0 -> class dyskinesia
    df.loc[df['PD3'].values > 0, ['4classPD3']] = 'dyskinesia'
    # If PD3>=0 (also equal to 0!) and symptoms "dyskinesia" --> class dyskinesia
    #df['4classPD3'][np.logical_and(np.char.count(df['symptoms'].values.astype('string'),'dyskinesia') > 0, df['PD3'].values >= 0)] = 'dyskinesia'
    
    # BRADYKINESIA
    # Generally, PD3<0, irregardless of symptoms --> class bradykinesia
    df.loc[df['PD3'].values < 0, ['4classPD3']] = 'bradykinesia'
    
    # TREMOR    
    # PD3<=0 and symptoms "tremor" --> class tremor
    df.loc[np.logical_and(np.char.count(df['symptoms'].values.astype('string'),'tremor') > 0, df['PD3'].values <= 0), ['4classPD3']] = 'tremor'
    # "RT" symptoms are resting tremor -> class "tremor"
    df.loc[np.logical_and(np.char.count(df['symptoms'].values.astype('string'),'RT') > 0, df['PD3'].values <= 0), ['4classPD3']] = 'tremor'
    
    # BALANCED
    # generally, if PD3==0 -> class "balanced"    
    df.loc[df['PD3'].values == 0,['4classPD3']] = 'balanced'
    # PD3==0 and symptoms "none" --> class "balanced"
    #df['4classPD3'][np.logical_and(df['symptoms'].values == 'none', df['PD3'].values == 0)] = 'balanced'
    
    # PD3=='x' --> class "?" (missing value)
    # (note: if there aren't any "x" values in the Excel column "PD3", this column will evaluate to a float64 vector 
    # and throw an error in the string comparison, so I first check for that)
    if df['PD3'].values.dtype.type is not np.float64:
        df.loc[df['PD3'].values == 'x', ['4classPD3']] = '?'
    return df


# FELIX
def setDataParameters(dataCols, labelCols):
    parameters = {}
    parameters['dataCols'] = dataCols
    parameters['labelCols'] = labelCols
    return parameters


def main():
    # obsolete main function, instead see sample script for data import
    
    # read in all data at once and store in list "pandata"
    # parameters (machine-dependent) for allocating a list of expert and wristband files 
    dataroot = 'path/to/data/root/folder'
    pattern = 'PT17.0*'  # only Parkinson patients, not healthy controls (those would be 'PT17.1*')
    dataFileTable = collectDataAndExpertFiles(dataroot, pattern=pattern)
    pandata = []
    for idx in range(4):  #(0,dataFileTable.shape[0]):
        dataBand, dataExperts = readSubjectDataAll(dataFileTable['ExpertFiles'][idx],dataFileTable['WristbandFiles'][idx])
        #testCheckClasses = dataExperts.loc[:,['PD3','symptoms','4classPD3']]
        try:
            dataMerged = synchronizeDataAndLabels(dataBand, dataExperts)
            pandata.append(dataMerged)
            print 'Merge of %d successful.' % idx
        except:
            #print "Unexpected error:", sys.exc_info()[0]
            print 'Merge of %d (%s) had a problem !!!' % (idx, dataFileTable['SubjectID'][idx])
            continue
    return pandata

# run main (only if it is the running file, not if it is being imported into another python file)
if __name__ == "__main__":
    #pandata = main()
    pass    
    
