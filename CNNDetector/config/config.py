# -*- coding: utf-8 -*-

class Config(object):
    def __init__(self):
        # Path
        self.root = '/home/yang/project/CNNDetector/CNNDetector'
        self.data_foldername = self.root + '/data'

        # Detection parameters
        self.resample = 100       # Hz
        self.winsize = 2001       # window length  20s * 100Hz
        self.winlag = 401         # step length    4s * 100Hz
        self.prob = 0.5
        self.group_num_thrd = 1   
        self.bandpass = [4, 15]   # bandpass 4-15 Hz
        self.plot = True          # plot detection results or not
        self.cut = True           # cut detection results or not
        self.new_scan = True      # If False, run the program from the breakpoint

        # Training parameters
        self.iteration = 3000
        self.learning_rate = 0.001
        self.cnn_pos_batch_num = 16
        self.cnn_neg_file_batch_num = 16
