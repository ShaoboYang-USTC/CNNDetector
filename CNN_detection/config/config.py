# -*- coding: utf-8 -*-

class Config(object):
    def __init__(self):
        # Path
        self.root = '/home/yang/project/CNN_Earthquake_Detection/CNN_detection'
        self.data_foldername = self.root + '/data'

        # Detection parameters
        self.resample = 100       # Hz
        self.winsize = 2001       # window length  10s * 200Hz
        self.winlag = 401         # step length    2s * 200Hz
        self.prob = 0.5
        self.bandpass = [1,40]    # bandpass 1-40Hz
        self.plot = True          # plot detection result or not

        # Training parameters
        self.iteration = 3000
        self.learning_rate = 0.001
        self.cnn_pos_batch_num = 16
        self.cnn_neg_file_batch_num = 16


