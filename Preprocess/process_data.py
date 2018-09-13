#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:02:25 2017

@author: Junwoo Suh

Functions:
    check_memory(gb):
        exit the program if the available memory is less than gb GB
    sig_corr(data, u_M, ind)
        

Class: raw_signal
    Functions:
        get_record_info():
            return sampling frequency, channel names and length of the data
        process_annot_n_header():
            add s_diff and e_diff attribute to the raw_signal
        process_record:
            add ch_info, data and time attribute to the raw_signal
        artifact_removal_ICA(visualize,r_thres):
            add corrected_data attribute to the raw_signal
            repeat ICA if r values are not satisfied
            also plots the result of artifact removal
    Attributes:
        record: path to the edf file
        annotation: path to the annotation file
        header: path to the header file
        s_diff: difference between the annotation and record at the start
        e_diff: difference between the annotation and record at the start
        ch_info: list of channel type(EEG, EOG, etc.) and channel index
        data: numpy array of the data (ch by sample)
        time: numpy array of the time (1 by sample)
        corrected_data: artifact removed data
        
"""
import sys
import mne
import numpy as np
from scipy.stats import pearsonr
import psutil  # For checking memory
from preprocess import io as in_out
from sklearn.decomposition import FastICA
import copy
import matplotlib.pyplot as plt

def check_memory(gb):
    # Check if there are sufficent memory
    memThreshold = gb * 1024 * 1024 * 1024 # x GB of minimum free memory
    """if psutil.virtual_memory().available < memThreshold:
        print("Warning: Free more memory before continuing")
        sys.exit()"""

def sig_corr(data, u_M):
    r = np.zeros((u_M.shape[1],2)) # r is x by 2 bc pearsonr is two-tailed
    for i in np.arange(0,u_M.shape[1]-1):
        r[i,:]=np.absolute(pearsonr(data,u_M[:,i])) # Two-tailed p-test.
    indx = np.argmax(r,axis=0)[np.argmax(np.amax(r,axis=0))] # Find the index of maximum r value

    return indx, r

def sc_buffer_cropping (data, sc_list, buffer_time):
    #Every sleep scores are intervaled at 30 sec
    buffer_time = buffer_time.split(':')
    #Convert time into an index
    buffer_sc = (int(buffer_time[0])*3600 + int(buffer_time[1])*60 + int(buffer_time[2]))/30
    #Initialize variables
    wake_count, p_sc, first_sleep_onset, lc_sc, fc_sc = 0, 0, 0, 0, 0
    sleep_onset, sleep_offset = 0, 0 
    for i, sc in enumerate(sc_list):
        # Check the sleeping status of the patient
        if p_sc == 0 and (sc == 1 or sc == 2 or sc == 3 or sc == 4):
            sleep_onset = 1
            sleep_offset = 0
        if (p_sc == 1 or p_sc == 2 or p_sc == 3 or p_sc == 4) and sc == 0:
            sleep_onset = 0
            sleep_offset = 1
            wake_count = 0

        # Find the index of sc where it has first sleep onset
        if sleep_onset and first_sleep_onset == 0:
            first_sleep_onset = 1 #This prevents future sleep onset from changing fc_sc
            fc_sc = (i - buffer_sc)*30
            if fc_sc < 0:
                fc_sc = 0

        if sleep_offset and sc == 0: # If patient has been sleeping and current state is sleeping
            wake_count = wake_count + 1
            if i == len(sc_list)-1: #The sleep lasted to the end of the data
                lc_sc = (wake_count - buffer_sc)*30
                if lc_sc < 0:
                    lc_sc = 0
        p_sc = sc
    return fc_sc, lc_sc

#%% #define raw_signal
class raw_signal:
    def __init__(self, record, annotation, header):
        # Three variables are initially stored: path for the record, annotation and header
        self.record = record
        self.annotation = annotation
        self.header = header

    def get_record_info(self,raw):
        # Get signal information
        sfreq = raw.info['sfreq']
        ch_names = raw.info['ch_names']
        data_size = raw.n_times
        raw.close()
        return sfreq, ch_names, data_size
    
    def process_annot_n_header(self):# Process annotation file
        #Read text based annotation file
        anot_start, anot_end, sc_list = in_out.read_annot(self.annotation)
        anot_end_sec = 3600*int(anot_end[0]) + 60*int(anot_end[1]) + int(anot_end[2])
        if int(anot_start[0]) > int(anot_end[0]):
            anot_start_sec = 3600*24 - (3600*int(anot_start[0]) + 60*int(anot_start[1]) + int(anot_start[2]))

        else:
            anot_start_sec = 3600*int(anot_start[0]) + 60*int(anot_start[1]) + int(anot_start[2])
        anot_tot = anot_end_sec + anot_start_sec
        pname = self.annotation.split('/')[-1].split('.')[0] # Patient name
        r_start, r_length = in_out.read_header(pname,self.header)
        
        #Find the difference between the record and annoation (in secs)
        r_tot = 3600*int(r_length[0]) + 60*int(r_length[1]) + float(r_length[2])

        # Find the starting time (in second) of the annotation file
        if int(anot_start[0]) > int(anot_end[0]) and int(r_start[0]) > int(anot_end[0]):
            # Case 1: annotation and recording happens before 0:00:00
            s_diff = ((3600*int(anot_start[0]) + 60*int(anot_start[1]) + float(anot_start[2])) - 
                  (3600*int(r_start[0]) + 60*int(r_start[1]) + float(r_start[2])))
        elif int(anot_start[0]) <= int(anot_end[0]) and int(r_start[0]) > int(anot_end[0]):
            # Case 2: annotation happens after 0:00:00 and recording happens before 0:00:00
            s_diff = ((3600*int(anot_start[0]) + 60*int(anot_start[1]) + float(anot_start[2])) + 
                  (3600*24 - (3600*int(r_start[0]) + 60*int(r_start[1]) + float(r_start[2]))))
        elif int(anot_start[0]) > int(anot_end[0]) and int(r_start[0]) <= int(anot_end[0]):
            # Case 3: annotation happens before 0:00:00 and recording happens after 0:00:00
            s_diff = (3600*24 - (3600*int(anot_start[0]) + 60*int(anot_start[1]) + float(anot_start[2])) + 
                  (3600*int(r_start[0]) + 60*int(r_start[1]) + float(r_start[2])))
        elif int(anot_start[0]) <= int(anot_end[0]) and int(r_start[0]) <= int(anot_end[0]):
            # Case 4: annotation and recording happens after 0:00:00
            s_diff = ((3600*int(anot_start[0]) + 60*int(anot_start[1]) + float(anot_start[2])) - 
                  (3600*int(r_start[0]) + 60*int(r_start[1]) + float(r_start[2])))
            
        # Calculate the difference near the end
        e_diff = r_tot - anot_tot - s_diff
        self.s_diff = s_diff
        self.e_diff = e_diff
        self.sc_list = sc_list
        print("<s_diff and e_diff added to the raw_signal>")

    def process_record(self, eeg_ch, emg_ch, ecg_ch, eog_ch,american,h_freq,buffer_time):
        print("<Processing the record>")
        # Check memory for 2GB before loading data
        check_memory(2)

        #Load the data
        raw = mne.io.read_raw_edf(self.record)
        raw.load_data()

        #Crop the data so that it only contains annotated parts
        fc_sc, lc_sc = sc_buffer_cropping (self.record, self.sc_list, buffer_time)
        c1, c2 = fc_sc + self.s_diff, lc_sc + self.e_diff

        if c1 > 0 and c2 > 0:
            raw.crop(tmin = c1, tmax = raw.times[-1] - c2)
        elif c1 < 0 and c2 > 0:
            raw.crop(tmin = 0, tmax = raw.times[-1] - c2)
            print("Warning: record starts later than annotation!! Check files")
        elif c1 > 0 and c2 < 0:
            raw.crop(tmin = c1, tmax = None)
            print("Warning: record ends earlier than annotation!! Check files")
        else:
            print("Warning: record is smaller than annotation!! Check files")

        #Label channels
        ch_t, ch_ind = [],[]
        sfreq, ch_names, data_size = self.get_record_info(raw)
        for i,ch in enumerate(ch_names):
            if ch in eeg_ch:
                ch_ind.append(i)
                ch_t.append('EEG')
            elif ch in ecg_ch:
                ch_ind.append(i)
                ch_t.append('ECG')
            elif ch in emg_ch:
                ch_ind.append(i)
                ch_t.append('EMG')
            elif ch in eog_ch:
                ch_ind.append(i)
                ch_t.append('EOG')
        pick_Ch = ch_ind

        # Drop channels that are not EEG, ECG and EOG
        drop_ch = set(np.arange(0,len(ch_names))) - set(pick_Ch)
        for ch in drop_ch:
            raw.drop_channels([ch_names[ch]])

        # Remove power-line noise with notch filtering
        if american:
            powerline = np.arange(60,sfreq/2,60) #American AC power at 60Hz
        else:
            powerline = np.arange(50,sfreq/2,50) #European AC power at 50Hz
        raw.notch_filter(powerline)
        raw.filter(None, h_freq, h_trans_bandwidth='auto', filter_length='auto', phase='zero')
        raw.close()
        self.ch_info = [ch_t,ch_ind]
        self.data, self.time = raw.get_data(picks=pick_Ch, start=0, return_times=True)
        print("<ch_info, data and time added to raw_signal>")


    def artifact_removal_ICA(self,visualize,r_thres):
        print("<Start ICA artifact removal>")
        self.data= self.data.T # Transpose matrix for FastICA
        
        r_min, BAD_ICA = 0, False
        while r_thres > r_min and not BAD_ICA:
            #Run FastICA
            ica = FastICA(n_components=self.data.shape[1])
            u_M = ica.fit_transform(self.data)
            w_M = ica.mixing_

            mix_ch_info = copy.deepcopy(self.ch_info[0])
            #Find EOG channel using Pearson Correlation
            ind = self.ch_info[1][self.ch_info[0].index('EOG')]
            EOG, rV = sig_corr(self.data[:,ind], u_M)
            if np.amax(rV) > r_min:
                r_min = np.amax(rV)
            if ind != EOG:
                mix_ch_info[EOG], mix_ch_info[ind] = 'EOG', 'EEG'
            
            #Find ECG channel using pearson Correlation
            ind = self.ch_info[1][self.ch_info[0].index('ECG')]
            ECG, rV = sig_corr(self.data[:,ind], u_M)
            if np.amax(rV) < r_min:
                r_min = np.amax(rV)
            if ind != ECG:
                mix_ch_info[ECG], mix_ch_info[ind] = 'ECG', 'EEG'
    
            #Find EMG channel using pearson Correlation
            ind = self.ch_info[1][self.ch_info[0].index('EMG')]
            EMG, rV = sig_corr(self.data[:,ind], u_M)
            if np.amax(rV) < r_min:
                r_min = np.amax(rV)
            if ind != EMG:
                mix_ch_info[EMG], mix_ch_info[ind] = 'EMG', 'EEG'
            if EOG == ECG or EOG == EMG or ECG == EMG:
                BAD_ICA = True

        # Zero-out the artifact channels
        inv_w_M = np.linalg.inv(w_M) # Inverse of mixing matrix
        u_corr_M = copy.deepcopy(u_M)
        u_corr_M[:,EOG] = np.zeros((u_M.shape[0]))
        u_corr_M[:,EMG] = np.zeros((u_M.shape[0]))
        u_corr_M[:,ECG] = np.zeros((u_M.shape[0]))
        self.corrected_data = np.dot(inv_w_M,u_corr_M.T)
        
        print("<corrected_data added to the raw_signal>")
        if visualize:
            #reassembled_data = np.dot(w_M,u_M.T).T # To check if ICA is working properly
            check_memory(0.5)
            print("plotting the signal because visualize=True")
            plt.figure()
            plt_time = self.time[:15000]
            groups = [self.data[:15000,:], u_M[:15000,:],self.corrected_data.T[:15000,:]]
            group_names = ['Observations (mixed signal)', 'ICA source signals',
                     'ICA recovered signals','Reassembled data (debuging)']
            sig_names = self.ch_info[0]
            for i, (group, g_names)in enumerate(zip(groups,group_names),1):
                plt.figure(i)
                plt.title(g_names)
                for ii, sig in zip(np.arange(0,8),group.T):
                    ax = plt.subplot(4,2,ii+1)
                    if i == 2:
                        ax.set_title(mix_ch_info[ii])
                    else:
                        ax.set_title(sig_names[ii])
                    plt.plot(plt_time,sig)
            plt.show()
        del self.data #Remove raw data to lighten up the memory
