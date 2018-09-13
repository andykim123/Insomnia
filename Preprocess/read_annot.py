#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:31:54 2017

@author: Junwoo Suh

This code accept .txt version of annotation file only.
The code generates 
"""

import os
from tkinter import Tk
from tkinter import filedialog
import pickle

#%% Read EDF file using Gui
root = Tk()
root.aname = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select EDF annotation file",filetypes = (("text files","*.txt"),("all files","*.*")))
root.header = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select EDF header file",filetypes = (("text files","*.txt"),("all files","*.*")))
root.destroy()
has_sleep_score = False # Bool for checking if data started
pname = root.aname.split('/')[-1].split('.')[0] # Patient name
#%% Read and record sleep score
with open(root.aname,"r") as input_file:
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
                sc_list.append(n_sc) # Record sleep Scores
            if delimited_line[time_ind] != '':
                line_buff = delimited_line
            else:
                continue
            

#%% Compute total annotation time
if delimited_line[time_ind] != '':
    anot_end = delimited_line[time_ind] # Find end time
else:
    anot_end = line_buff[time_ind] # Find end time

anot_start, anot_end = anot_start.split(":"), anot_end.split(":")

anot_end_sec = 3600*int(anot_end[0]) + 60*int(anot_end[1]) + int(anot_end[2])

if int(anot_start[0]) > int(anot_end[0]):
    anot_start_sec = 3600*24 - (3600*int(anot_start[0]) + 60*int(anot_start[1]) + int(anot_start[2]))

else:
    anot_start_sec = 3600*int(anot_start[0]) + 60*int(anot_start[1]) + int(anot_start[2])
anot_tot = anot_end_sec + anot_start_sec

#%% Get header of the file
pstart = False
pN = 0
with open(root.header, "r") as input_file:
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
            r_length = line.split(":")[1:4]
            r_length[2] = r_length[2].split('(')[0]
            break
#%% Find the difference between the record and annoation (in secs)
r_tot = 3600*int(r_length[0]) + 60*int(r_length[1]) + float(r_length[2])
#if anot_tot < r_tot: # Check if the annotation is shorter than the recoding


# Find the starting time (in second) of the annotation file

if int(anot_start[0]) > int(anot_end[0]) and int(r_start_time[0]) > int(anot_end[0]):
    # Case 1: annotation and recording happens before 0:00:00
    s_diff = ((3600*int(anot_start[0]) + 60*int(anot_start[1]) + float(anot_start[2])) - 
          (3600*int(r_start_time[0]) + 60*int(r_start_time[1]) + float(r_start_time[2])))
    
elif int(anot_start[0]) <= int(anot_end[0]) and int(r_start_time[0]) > int(anot_end[0]):
    # Case 2: annotation happens after 0:00:00 and recording happens before 0:00:00
    s_diff = ((3600*int(anot_start[0]) + 60*int(anot_start[1]) + float(anot_start[2])) + 
          (3600*24 - (3600*int(r_start_time[0]) + 60*int(r_start_time[1]) + float(r_start_time[2]))))
    
elif int(anot_start[0]) > int(anot_end[0]) and int(r_start_time[0]) <= int(anot_end[0]):
    # Case 3: annotation happens before 0:00:00 and recording happens after 0:00:00
    s_diff = (3600*24 - (3600*int(anot_start[0]) + 60*int(anot_start[1]) + float(anot_start[2])) + 
          (3600*int(r_start_time[0]) + 60*int(r_start_time[1]) + float(r_start_time[2])))
    
elif int(anot_start[0]) <= int(anot_end[0]) and int(r_start_time[0]) <= int(anot_end[0]):
    # Case 4: annotation and recording happens after 0:00:00
    s_diff = ((3600*int(anot_start[0]) + 60*int(anot_start[1]) + float(anot_start[2])) - 
          (3600*int(r_start_time[0]) + 60*int(r_start_time[1]) + float(r_start_time[2])))
    
# Calculate the difference near the end
e_diff = r_tot - anot_tot - s_diff

# Negative s_diff or e_diff means that annotation file is larger than that of recording
if s_diff < 0 and e_diff < 0:
    you_are_fucked_activated = True
else:
    you_are_fucked_activated = False    




#%% Read  Save the sleep score and time
with open('temp_annot.pickle', 'wb') as f:
    pickle.dump([s_diff, e_diff,you_are_fucked_activated],f)