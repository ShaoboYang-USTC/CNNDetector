
import os.path
import glob
import random
import numpy as np
# np.set_printoptions(threshold=np.nan)
import obspy
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from obspy.signal.filter import bandpass

from config.config import Config

# def data_preprocess(data, filter_model='bandpass'):
#     if filter_model == 'bandpass':
#         data = bandpass(data, freqmin=5, freqmax=15, df=200)

#     data = (data - np.mean(data)) / (np.max(np.absolute(data)) + 1)
#     data = np.absolute(data)
#     return data

class Reader(object):
    def __init__(self):
        self.config = Config()
        self.foldername = self.config.data_foldername
        self.winsize = self.config.winsize
        self.beforename = self.get_filename('Noise')
        self.microSeismic = self.get_filename('Events')
        # self.aftername = self.get_filename('after')
        #self.aftername = self.get_filename('detectingdata')
        # self.examplename = self.get_filename('example')
        # self.events_examplename = self.get_filename('events_example')

        # self.new_events = self.get_filename('new_events')

    def get_filename(self, dataset_type):
        filename_dir = os.path.join(self.foldername, dataset_type)
        filename_dict = dict()
        if os.path.exists(filename_dir):
            filename_list = os.listdir(filename_dir)
            for name in filename_list:
                EventfileDir = os.path.join(filename_dir, name)
                # SACfiles = os.listdir(EventfileDir)
                os.chdir(EventfileDir)
                for SACfilename in glob.glob('*E'):
                    E = os.path.join(EventfileDir, SACfilename[:-1]+'E')
                    N = os.path.join(EventfileDir, SACfilename[:-1]+'N')
                    Z = os.path.join(EventfileDir, SACfilename[:-1]+'Z')
                    if name not in filename_dict.keys():
                        filename_dict[name] = [[E],[N],[Z]]
                    else:
                        filename_dict[name][0].append(E)
                        filename_dict[name][1].append(N)
                        filename_dict[name][2].append(Z)
        else:
            print('{} is not exist.'.format(filename_dir))
            return None

        filename_list = list(filename_dict.values()) #filename_dict.values每个值有三分量 总数为事件数目
        # sort
        for i in range(len(filename_list)):
            for j in range(3):
                filename_list[i][j].sort()
        return filename_list

    def read_sac_data(self, file_list, normalize=True):

        # print(file_list)
        file_num = len(file_list)
        batch_data = list()

        traces = []
        for i in range(file_num):
            traces.append([])
            file_len = len(file_list[i][0])
            for j in range(3):
                traces[i].append([])
                # print(traces)
                for k in range(file_len):
                    # traces[i][j].append([])
                    tmp_filename = os.path.join(self.foldername, file_list[i][j][k])
                    tmp_trace = obspy.read(tmp_filename)[0]

                    traces[i][j].append(tmp_trace)

        trace_len = []
        for i in range(file_num):
            file_len = len(file_list[i][0])
            trace_len.append(traces[i][0][0].stats.npts)

            for j in range(3):
                for k in range(file_len):
                    traces[i][j][k] = list(traces[i][j][k].data)

        tmp_data = []
        for i in range(file_num):
            file_len = len(file_list[i][0])
            tmp_data.append([])
            for j in range(3):
                if trace_len[i] < self.winsize:
                    traces[i][j] = np.concatenate((traces[i][j], np.zeros([file_len, self.winsize - trace_len[i]])), axis=1)

            traces[i] = np.array(traces[i])
            tmp_data[i] = traces[i][:, :, :self.winsize]
            #print(tmp_data)

            if normalize:
                for j in range(3):
                    for k in range(file_len):
                        mean_data = np.mean(tmp_data[i][j][k])
                        abs_tmp_data = np.absolute(tmp_data[i][j][k])
                        # tmp_data = tmp_data / (np.max(abs_tmp_data) + np.array([1, 1, 1]))
                        tmp_data[i][j][k] = (tmp_data[i][j][k] - mean_data) / (np.max(abs_tmp_data) + 0.1)

            # batch_data.append(np.array([tmp_data]))
        # batch_data size: n*3*3*winsize
        #tmp_data = np.array(tmp_data)
        batch_data = np.array(tmp_data)
        batch_data=batch_data.transpose((0,2,3,1))

        return batch_data

    # The input size of each batch bust be the same
    def process_data_file(self, old_batch_file, num):
        #print(old_batch_file[0:2])
        for i in range(len(old_batch_file)):
            old_batch_file[i] = np.array(old_batch_file[i])
            old_batch_file[i] = old_batch_file[i].T
            np.random.shuffle(old_batch_file[i])
            old_batch_file[i] = list(old_batch_file[i])
            old_batch_file[i] = old_batch_file[i][0:num]
            old_batch_file[i] = np.array(old_batch_file[i])
            old_batch_file[i] = old_batch_file[i].T
            old_batch_file[i] = list(old_batch_file[i])

        new_batch_file = np.array(old_batch_file)
        return new_batch_file

    def get_cnn_batch_data(self, data_type):
        train_pos_file_name, test_pos_file_name = train_test_split(self.microSeismic,
                                                                   test_size=0.1, random_state = 0)
        train_neg_file_name, test_neg_file_name = train_test_split(self.beforename,
                                                                   test_size=0.1, random_state = 0)
        train_pos_num = len(train_pos_file_name)
        train_neg_num = len(train_neg_file_name)
        pos_batch_num = self.config.cnn_pos_batch_num
        neg_file_batch_num = self.config.cnn_neg_file_batch_num

        # neg_file_per_num = self.config.cnn_neg_file_per_num

        def get_batch():
            pos_batch_i = 0
            neg_batch_i = 0
            while True:
                if pos_batch_i == 0:
                    random.shuffle(train_pos_file_name)

                if neg_batch_i == 0:
                    random.shuffle(train_neg_file_name)

                if pos_batch_i + pos_batch_num > train_pos_num:
                    pos_batch_i = train_pos_num - pos_batch_num

                if neg_batch_i + neg_file_batch_num > train_neg_num:
                    neg_batch_i = train_neg_num - neg_file_batch_num

                pos_batch_file = train_pos_file_name[pos_batch_i: pos_batch_i + pos_batch_num]
                neg_batch_file = train_neg_file_name[neg_batch_i: neg_batch_i + neg_file_batch_num]

                each_num = []
                for i in range(len(pos_batch_file)):
                    each_num.append(len(pos_batch_file[i][0]))
                for i in range(len(neg_batch_file)):
                    each_num.append(len(neg_batch_file[i][0]))
                min_num = min(each_num)

                # random select station number:
                sta_num = []
                for i in range(4, min_num + 1):
                    sta_num.append(i)
                num = random.sample(sta_num, 1)[0]

                pos_batch_file = self.process_data_file(pos_batch_file, num = num)
                neg_batch_file = self.process_data_file(neg_batch_file, num = num)

                pos_x = self.read_sac_data(pos_batch_file)
                batch_y = np.ones(len(pos_x), dtype=int)
                batch_x = pos_x

                neg_x = self.read_sac_data(neg_batch_file)

                batch_y = np.concatenate((batch_y, np.zeros(len(neg_x), dtype=int)))
                batch_x = np.concatenate((batch_x, neg_x))

                yield batch_x, batch_y

                pos_batch_i += pos_batch_num
                neg_batch_i += neg_file_batch_num

                if pos_batch_i == train_pos_num:
                    pos_batch_i = 0
                if neg_batch_i == train_neg_num:
                    neg_batch_i = 0

        batch=get_batch()
        if data_type == 'train':
            return next(batch)

        if data_type == 'test':
            #sample_pos_file_name = random.sample(test_pos_file_name, 50)
            sample_pos_file_name = self.process_data_file(test_pos_file_name, num = 5)
            sample_neg_file_name = self.process_data_file(test_neg_file_name, num = 5)
            #sample_neg_file_name = random.sample(test_neg_file_name, 20)
            #print(sample_neg_file_name[0:9])
            #print(sample_pos_file_name[0:9])

            test_pos_x = self.read_sac_data(sample_pos_file_name)
            test_pos_y = np.ones(len(test_pos_x), dtype=int)
            test_neg_x = self.read_sac_data(sample_neg_file_name)
            test_y = np.concatenate((test_pos_y, np.zeros(len(test_neg_x), dtype=int)))
            test_x = np.concatenate((test_pos_x, test_neg_x))
            return test_x, test_y

if __name__ == '__main__':
    pass
