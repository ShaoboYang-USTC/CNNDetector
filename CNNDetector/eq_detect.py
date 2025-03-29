"""
 Author: Shaobo Yang
 Time:   10/29/2020
 Email:  yang0123@mail.ustc.edu.cn
"""

import csv
import os
import glob
import subprocess
import numpy as np
import obspy
from obspy.signal.filter import bandpass
from obspy import UTCDateTime
import datetime
# from csvreader import csvreader
import time
from functools import reduce
# np.set_printoptions(threshold=np.nan)

from config.config import Config
from reader.reader import Reader


def data_preprocess(data, isabs=False):

    # normalization
    max_amp = np.max(np.absolute(data))
    if max_amp != 0:
        data = (data - np.mean(data)) / max_amp

    if isabs:
        data = np.absolute(data)
    return data

def main(model_name, new_scan=False, preprocess=True):
    config = Config()
    plot = config.plot
    cut = config.cut
    bandpass = config.bandpass
    resample = config.resample

    # read data folders
    file_list = os.listdir(config.data_foldername+'/after')
    file_list.sort()
    if new_scan == True:
        print('start new scan!')
        start_point = 0
        event_num = 0

        try:
            os.system('rm -rf %s/event_detect/detect_result/cut/*'%config.root)
            os.system('rm -rf %s/event_detect/detect_result/png/*'%config.root)
            os.system('rm -rf %s/event_detect/detect_result/png2/*'%config.root)
            os.system('rm -rf %s/event_detect/detect_result/cnn/*.csv'%config.root)
        except:
            pass
        # file_list_len = len(file_list)
    else:
        with open(config.root + '/event_detect/detect_result/' + model_name + '/checkpoint') as file:
            start_point = int(file.readline())
            event_num = int(file.readline())
            file_list = file_list[start_point:]
            # file_list_len = len(file_list)
            print('restart from {}'.format(file_list[0]))
    
    # load CNN model
    if model_name == 'cnn':
        from cnn import CNN
        import tensorflow as tf
        from tflib.models import Model

        model = CNN()
        sess = tf.Session()
        saver, global_step = Model.continue_previous_session(sess,
                                                             model_file='cnn',
                                                             ckpt_file=config.root + '/event_detect/saver/cnn/checkpoint')

    # read group info
    group = []
    with open(config.root+'/config/group_info', 'r') as f:
        for line in f.readlines():
            if line != '\n':
                if line[0] == '#':
                    group.append([])
                else:
                    group[-1].append(line.split()[0])
    # read data & detect eq
    for file in file_list:
        sac_file_name = [[], [], []]
        all_group_sta_num = [0]*len(group)
        path = os.path.join(config.data_foldername+'/after', file)
        begin = datetime.datetime.now()
        group_E = [[] for _ in range(len(group))]
        group_N = [[] for _ in range(len(group))]
        group_Z = [[] for _ in range(len(group))]
        print('Start reading data: %s.'%file)
        max_start_time = None 
        min_end_time = None 
        for i in range(len(group)):
            for sta in group[i]:
                if len(glob.glob(path+'/'+'*'+sta+'.*')) == 3:
                    all_group_sta_num[i] += 1
                    sacfile_E = glob.glob(path+'/'+'*'+sta+'.*'+'E')[0]
                    sacfile_N = glob.glob(path+'/'+'*'+sta+'.*'+'N')[0]
                    sacfile_Z = glob.glob(path+'/'+'*'+sta+'.*'+'Z')[0]
                    sac_file_name[0].append(sacfile_E.split('/')[-1])
                    sac_file_name[1].append(sacfile_N.split('/')[-1])
                    sac_file_name[2].append(sacfile_Z.split('/')[-1])
                    group_E[i].append(obspy.read(sacfile_E))
                    group_N[i].append(obspy.read(sacfile_N))
                    group_Z[i].append(obspy.read(sacfile_Z))
                    start_time = group_Z[i][-1][0].stats.starttime
                    end_time = group_Z[i][-1][0].stats.endtime
                    if max_start_time:
                        max_start_time = max_start_time if max_start_time > start_time else start_time
                        min_end_time = min_end_time if min_end_time < end_time else end_time
                    else:
                        max_start_time = start_time
                        min_end_time = end_time
        if max_start_time >= min_end_time:
            continue
        flatten_group_E = [st for each_group in group_E for st in each_group]
        flatten_group_N = [st for each_group in group_N for st in each_group]
        flatten_group_Z = [st for each_group in group_Z for st in each_group]
        st_E = reduce(lambda st1, st2:st1+st2, flatten_group_E)
        st_N = reduce(lambda st1, st2:st1+st2, flatten_group_N)
        st_Z = reduce(lambda st1, st2:st1+st2, flatten_group_Z)
        st_all = st_E + st_N + st_Z
        st_all = st_all.slice(max_start_time, min_end_time)
        all_sta_num = len(flatten_group_Z)
        if resample:
            st_all = st_all.resample(sampling_rate = resample)
        if bandpass:
            st_all = st_all.filter('bandpass',freqmin=bandpass[0],freqmax=bandpass[1],corners=4,zerophase=True)
        endtime = st_all[0].stats.endtime

        start_flag = -1
        end_flag = -1
        event_list = []
        confidence_total = {}
        start_total = []
        end_total = []
        pos_num_total = []
        samples = 1.0/st_all[0].stats.delta
        # npts = st_all[0].stats.npts
        print('Finish reading data.')
        
        print('Start detection.')
        slices = st_all.slide(window_length=(config.winsize-1)/samples,
                                        step=config.winlag/samples)
        windowed_st = next(slices)      # skip the first window to avoid the taper influence
        for windowed_st in slices:
            cur_sta = 0
            len_group_conf = 0
            group_class, group_conf = [], []
            # windowed_E = windowed_st[:all_sta_num]
            # windowed_N = windowed_st[all_sta_num:2*all_sta_num]
            # windowed_Z = windowed_st[2*all_sta_num:]
            start = len(windowed_st)/3*2
            end = len(windowed_st)
            group_max_conf = 0
            for i in range(len(group)):
                data_input = [[],[],[]]
                group_sta_num = all_group_sta_num[i]
                if group_sta_num > 0:
                    for j in range(cur_sta, cur_sta+group_sta_num):
                        if len(windowed_st[j].data) < config.winsize:
                            windowed_st[j].data = np.concatenate([windowed_st[j].data,np.zeros(config.winsize-len(windowed_st[j].data))])
                        data_input[0].append(windowed_st[j].data[:config.winsize])
                        # print(j, windowed_st[j])
                    for j in range(all_sta_num+cur_sta, all_sta_num+cur_sta+group_sta_num):
                        if len(windowed_st[j].data) < config.winsize:
                            windowed_st[j].data = np.concatenate([windowed_st[j].data,np.zeros(config.winsize-len(windowed_st[j].data))])
                        data_input[1].append(windowed_st[j].data[:config.winsize])
                        # print(j, windowed_st[j])
                    for j in range(2*all_sta_num+cur_sta, 2*all_sta_num+cur_sta+group_sta_num):
                        if len(windowed_st[j].data) < config.winsize:
                            windowed_st[j].data = np.concatenate([windowed_st[j].data,np.zeros(config.winsize-len(windowed_st[j].data))])
                        data_input[2].append(windowed_st[j].data[:config.winsize])
                        # print(j, windowed_st[j])
                    plot_b = 2*all_sta_num+cur_sta
                    plot_e = 2*all_sta_num+cur_sta+group_sta_num
                    cur_sta += group_sta_num
            
                    if preprocess:
                        for i in range(3):
                            for j in range(group_sta_num):
                                data_input[i][j] = data_preprocess(data_input[i][j])
                    data_input = np.array(data_input)

                    if len(data_input[0][0]) < config.winsize:
                        concat = np.zeros([3, group_sta_num, config.winsize - len(data_input[0][0])])
                        data_input = np.concatenate([data_input,concat],axis=2)
                    else:
                        data_input = data_input[:, :, :config.winsize]
                    data_input = data_input.transpose((1,2,0))

                    j = 0
                    while j < len(data_input):
                        if np.max(data_input[j]) == 0 or np.isnan(np.max(data_input[j])):
                            data_input = np.delete(data_input, j, axis = 0)
                        else:
                            j += 1
                    
                    if len(data_input) >= 3:
                        len_group_conf += 1
                        class_pred, confidence = model.classify(sess=sess, input_=[data_input])
                        group_class.append(class_pred)
                        group_conf.append(confidence[0])
                        if confidence[0] > group_max_conf:
                            start = plot_b
                            end = plot_e
                            group_max_conf = confidence[0]
                    else:
                        group_class.append(0)
                        group_conf.append(0)
                else:
                    group_class.append(0)
                    group_conf.append(0)
            group_conf.sort(reverse=True)

            # consider the result of multiple groups
            pos_num = 0
            for each in group_class:
                if each == 1:
                    pos_num += 1
            if pos_num >= config.group_num_thrd:
                class_pred = 1
            else:
                class_pred = 0

            confidence = sum(group_conf)/len_group_conf if len_group_conf else 0

            # calculate the window range
            if class_pred == 1:
                confidence_total[confidence] = [group_max_conf, start, end]
                start_total.append(windowed_st[0].stats.starttime)
                end_total.append(windowed_st[0].stats.endtime)
                pos_num_total.append(pos_num)

                if start_flag == -1:
                    start_flag = windowed_st[0].stats.starttime
                    end_flag = windowed_st[0].stats.endtime
                else:
                    end_flag = windowed_st[0].stats.endtime
            print("{} {} {} {} {:.8f} {:.8f}".format(class_pred,start_flag,end_flag, 
                  windowed_st[0].stats.starttime,confidence,group_conf[min(config.group_num_thrd-1, 
                                                                           len(group_conf)-1)]))

            if class_pred == 0 and start_flag != -1:  #end_flag < windowed_st[0].stats.starttime:
                confidence = max(list(confidence_total.keys()))
                # for j in range(len(confidence_total)):
                #     if confidence == confidence_total[j]:
                #         break
                # start_local = start_total[j]
                # end_local = end_total[j]
                # event = [file, start_flag, end_flag,
                #          confidence, start_local, end_local]
                group_max_conf = confidence_total[confidence][0]
                start = confidence_total[confidence][1]
                end = confidence_total[confidence][2]
                event = [file, start_flag, end_flag, confidence, \
                    max(pos_num_total), start, end, group_max_conf]

                confidence_total={}
                start_total = []
                end_total = []
                pos_num_total = []

                event_list.append(event)
                #print(event_list)

                start_flag = -1
                end_flag = -1

            if class_pred == 1 and end_flag+config.winlag/samples >= endtime:
                confidence = max(list(confidence_total.keys()))
                # for j in range(len(confidence_total)):
                #     if confidence == confidence_total[j]:
                #         break
                # start_local = start_total[j]
                # end_local = end_total[j]
                # event = [file.split('/')[-2], start_flag, endtime,
                #          confidence, start_total, end_total]
                group_max_conf = confidence_total[confidence][0]
                start = confidence_total[confidence][1]
                end = confidence_total[confidence][2]
                event = [file, start_flag, endtime, confidence, \
                    max(pos_num_total), start, end, group_max_conf]

                event_list.append(event)
                start_flag = -1
                end_flag = -1

        if event_list:
            new_event_list = [event_list[0]]
            for i in range(1, len(event_list)):
                if event_list[i][1] > new_event_list[-1][1] and \
                event_list[i][1] < new_event_list[-1][1] + 1000*windowed_st[0].stats.delta:
                # if event_list[i][1] > new_event_list[-1][1] and event_list[i][1] < new_event_list[-1][2]:
                    new_event_list[-1][2] = event_list[i][2]
                else:
                    new_event_list.append(event_list[i])
            # Add index
            for i in range(len(new_event_list)):
                event_num += 1
                new_event_list[i].insert(0, event_num)
        else:
            new_event_list = []

        # write event list
        if len(new_event_list) != 0:
            with open(config.root + '/event_detect/detect_result/' + model_name + '/events_list.csv', mode='a', newline='') as f:
                csvwriter = csv.writer(f)
                for event in new_event_list:
                    csvwriter.writerow(event)
                f.close()

        if plot:
            print('Plot detected events.')
            for event in new_event_list:
                plot_traces = st_Z
                event_num, _, start_flag, end_flag, confidence, pos_num, start, end, group_max_conf = event
                name = config.root + '/event_detect/detect_result/png/' \
                        + str(int(event_num)) + '_' + str(confidence)[:4] + '.png'
                plot_traces.plot(starttime=start_flag, endtime=end_flag, size=(800, 800),
                                automerge=False, equal_scale=False, linewidth=0.8, outfile=name)

                plot_traces2 = st_all[start:end]
                name2 = config.root + '/event_detect/detect_result/png2/' \
                        + str(int(event_num)) + '_' + str(group_max_conf)[:4] + '.png'
                plot_traces2.plot(starttime=start_flag, endtime=end_flag, size=(800, 800),
                                automerge=False, equal_scale=False, linewidth=0.8, outfile=name2)

        ## cut use Obspy, processed data
        # if cut:
        #     print('Cut detected events.')
        #     for event in new_event_list:
        #         event_num, _, start_flag, end_flag, confidence, pos_num, start, end, group_max_conf = event
        #         slice_E = st_E.slice(start_flag, end_flag)
        #         slice_N = st_N.slice(start_flag, end_flag)
        #         slice_Z = st_Z.slice(start_flag, end_flag)
        #         save_path = config.root + '/event_detect/detect_result/cut/' \
        #                 + str(int(event_num)) + '_' + str(confidence)[:4]
        #         os.system('mkdir %s'%save_path)
        #         for i in range(len(slice_E)):
        #             slice_E[i].write(save_path+'/'+sac_file_name[0][i], format='SAC')
        #             slice_N[i].write(save_path+'/'+sac_file_name[1][i], format='SAC')
        #             slice_Z[i].write(save_path+'/'+sac_file_name[2][i], format='SAC')

        ## cut use SAC, raw data
        if cut:
            print('Cut detected events.')
            kzdate = os.popen('saclst kzdate f %s'%sacfile_Z).read().split()[1]
            kztime = os.popen('saclst kztime f %s'%sacfile_Z).read().split()[1]
            kzdate = kzdate.replace('/', '-')
            kzdatetime = UTCDateTime(f'{kzdate}T{kztime}')
            for event in new_event_list:
                event_num, _, start_flag, end_flag, confidence, pos_num, start, end, group_max_conf = event
                save_path = config.root + '/event_detect/detect_result/cut/' \
                    + str(int(event_num)) + '_' + str(confidence)[:4] + '/'
                os.system('mkdir %s'%save_path)
                cut_b = start_flag - kzdatetime
                cut_e = end_flag - kzdatetime
                # cut_b = 60*60*int(start_flag.hour) + 60*int(start_flag.minute) + float(start_flag.second)
                # cut_e = 60*60*int(end_flag.hour) + 60*int(end_flag.minute) + float(end_flag.second)
                ## SAC
                os.putenv("SAC_DISPLAY_COPYRIGHT","0")
                p=subprocess.Popen(['sac'],stdin=subprocess.PIPE)

                s=''
                
                s+="cut %s %s \n"%(cut_b, cut_e)
                s+="r %s/* \n"%(config.data_foldername+'/after/'+file)
                s+="w dir %s over \n"%(save_path)
                s+="quit \n"

                p.communicate(s.encode())

        start_point += 1
        with open(config.root + '/event_detect/detect_result/' + model_name + '/checkpoint', mode='w') as f:
            f.write(str(start_point) + '\n')
            f.write(str(event_num))
            end = datetime.datetime.now()
            print('{} completed, num {}, time {}.'.format(file, start_point, end-begin))
            print('Checkpoint saved.')

if __name__ == '__main__':
    main('cnn', new_scan=Config().new_scan)
