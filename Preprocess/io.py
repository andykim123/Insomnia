#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 01:48:56 2017

@author: Junwoo Suh

Deals with input and output of the data
"""

import os
from tkinter import Tk
from tkinter import filedialog
import preprocess.process_data as data


def open_data():
    # Read EDF file using Gui
    root = Tk()
    root.fname = filedialog.askopenfilename(initialdir=os.getcwd(),
                                            title="Select EDF data file",
                                            filetypes=(("EDF files", "*.edf"),
                                                       ("all files", "*.*")))
    root.aname = filedialog.askopenfilename(initialdir=os.getcwd(),
                                            title="Select EDF annotation file",
                                            filetypes=(("text files", "*.txt"), ("all files", "*.*")))
    root.header = filedialog.askopenfilename(initialdir=os.getcwd(),
                                             title="Select EDF header file",
                                             filetypes=(("text files", "*.txt"), ("all files", "*.*")))
    root.destroy()
    return data.raw_signal(root.fname, root.aname, root.header)

def read_annot(annot):
    # Read annotation file in text form
    has_sleep_score = False
    with open(annot,"r") as input_file:
        sc_list = []
        for i, line in enumerate(input_file):
            delimited_line = line.split("\t") # .txt file is delimited with tab
            if "Sleep Stage" in delimited_line : # Actual data starts after this line
                has_sleep_score = True
                data_start = i + 1 # Mark the line that the data starts
                sc_ind = delimited_line.index('Event')
                time_ind = delimited_line.index('Time [hh:mm:ss]')
                continue # Skip the line that contains names
            if has_sleep_score:
                if i == data_start:
                    anot_start = delimited_line[time_ind] # Find start time
                
                event = delimited_line[sc_ind].split("-") # event is used to remove CAP related events
                if event[0] == "SLEEP": # Get only sleep related events
                    sc = delimited_line[0]
                    if sc == 'W':
                        n_sc = 0
                    elif sc == 'R':
                        n_sc = 1
                    elif sc == 'S1':
                        n_sc = 2
                    elif sc == 'S2':
                        n_sc = 3
                    elif sc == 'S3':
                        n_sc = 4
                    else:
                        n_sc = 5
                    sc_list.append(n_sc)  # Record sleep Scores
                if delimited_line[time_ind] != '':
                    line_buff = delimited_line
                else:
                    continue
        if delimited_line[time_ind] != '':
            anot_end = delimited_line[time_ind] # Find end time
        else:
            anot_end = line_buff[time_ind] # Find end time
        
        anot_start, anot_end = anot_start.split(":"), anot_end.split(":")
    return anot_start, anot_end, sc_list
h_freq = 60 # Just a testing value. Can try different number under Nyquist Frequency

def read_header(pname,header):
    #Read the header file in text form
    pstart, pN = False, 0
    with open(header, "r") as input_file:
        for ii,line in enumerate(input_file):
            delimited_line = line.split('.')
            if delimited_line[0] == pname:
                pstart = True
                pN = ii
                continue
            # If the patient id matches get next two lines
            if pstart and ii == pN+ 1: 
                r_start_time = line[line.find("[")+1:line.find("]")].split(' ')[0].split(":")
            if pstart and ii == pN + 2:
                r_length = line.split(":")[1:4] #This is needed due to data formats
                r_length[2] = r_length[2].split('(')[0]
                break
    return r_start_time, r_length
