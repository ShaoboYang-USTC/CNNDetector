import csv
import os

import numpy as np
import obspy
from obspy.signal.filter import bandpass
import datetime
# from csvreader import csvreader
from time import time
np.set_printoptions(threshold=np.nan)

from config.config import Config
from reader.reader import Reader


def data_preprocess(data, isabs=False):

    # normalization
    data = data * 1e+3
    data = (data - np.mean(data)) / (np.max(np.absolute(data)) + 0.00001)

    # print(np.max(data))
    if isabs:
        data = np.absolute(data)
    return data

def main(model_name, new_scan=False, preprocess=True):
    reader = Reader()
    config = Config()
    plot = config.plot
    bandpass = config.bandpass
    resample = config.resample

    confidence0=[]
    plot_num = 0
    reader.aftername.sort()

    if new_scan == True:
        print('start new scan!')
        file_list = reader.aftername

        start_point = 0
    else:
        with open(config.root + '/event_detect/detect_result/' + model_name + '/checkpoint') as file:
            start_point = int(file.readline())
            file_list = reader.aftername[start_point:]
            print('restart from {}'.format(file_list[0]))

    if model_name == 'cnn':
        from cnn import CNN
        import tensorflow as tf
        from tflib.models import Model

        model = CNN()
        sess = tf.Session(config=tf.ConfigProto(device_count={"CPU":20},inter_op_parallelism_threads=0,intra_op_parallelism_threads=0))
        saver, global_step = Model.continue_previous_session(sess,
                                                             model_file='cnn',
                                                             ckpt_file=config.root + '/event_detect/saver/cnn/checkpoint')
    file_list_len = len(file_list)
    # print(file_list_len)

    try:
        os.system('rm -rf %s/event_detect/detect_result/png/*'%config.root)
        os.system('rm -rf %s/event_detect/detect_result/cnn/*.csv'%config.root)
    except:
        pass

    for file in file_list:
        file=np.array(file)
        #print(file)
        #file=file.T
        #np.random.shuffle(file)  #random
        #file=file.T

        #print(file,'\n')
        begin = datetime.datetime.now()
        if plot:
            plot_traces = obspy.read(file[2][0]) #Z component
        sta_num = len(file[0])
        trace_len = []

        for i in range(3):
            for j in range(sta_num):
                trace_len.append(obspy.read(file[i][j])[0].stats.npts)
        max_len = max(trace_len)

        for i in range(3):
            for j in range(sta_num):        # station number
                each_tr = obspy.read(file[i][j])
                if each_tr[0].stats.npts < max_len:
                    zero = np.zeros(max_len-each_tr[0].stats.npts)
                    each_tr[0].data = np.concatenate([each_tr[0].data,zero])
                if i==j==0:
                    traces = each_tr
                else:
                    traces=traces + each_tr
                if i == 2:
                    if j == 0:
                        pass
                    else:
                        plot_traces = plot_traces + each_tr

        if plot:
            if resample:
                plot_traces = plot_traces.resample(sampling_rate=resample)
            plot_traces = plot_traces.filter('bandpass',freqmin=bandpass[0],freqmax=bandpass[1],corners=4,zerophase=True)
        
        if resample:
            traces = traces.resample(sampling_rate=resample)
        traces = traces.filter('bandpass',freqmin=bandpass[0],freqmax=bandpass[1],corners=4,zerophase=True)
        starttime = traces[0].stats.starttime;
        endtime = traces[0].stats.endtime;
        #print(traces)

        start_flag = -1
        end_flag = -1
        event_list = []
        confidence_total=[]
        start_total=[]
        end_total=[]
        samples_trace= 1.0/traces[0].stats.delta;
        npts = traces[0].stats.npts


        for windowed_st in traces.slide(window_length=(config.winsize-1)/samples_trace,
                                        step=config.winlag / samples_trace):
            data_input = [[],[],[]]

            for j in range(sta_num):
                if len(windowed_st[j].data) < config.winsize:
                    windowed_st[j].data = np.concatenate([windowed_st[j].data,np.zeros(config.winsize-len(windowed_st[j].data))])
                data_input[0].append(windowed_st[j].data[:config.winsize])
            for j in range(sta_num,2*sta_num):
                if len(windowed_st[j].data) < config.winsize:
                    windowed_st[j].data = np.concatenate([windowed_st[j].data,np.zeros(config.winsize-len(windowed_st[j].data))])
                data_input[1].append(windowed_st[j].data[:config.winsize])
            for j in range(2*sta_num,3*sta_num):
                if len(windowed_st[j].data) < config.winsize:
                    windowed_st[j].data = np.concatenate([windowed_st[j].data,np.zeros(config.winsize-len(windowed_st[j].data))])
                data_input[2].append(windowed_st[j].data[:config.winsize])

            if model_name == 'cnn':

                if preprocess:
                    for i in range(3):
                        for j in range(sta_num):
                            data_input[i][j] = data_preprocess(data_input[i][j])

                data_input=np.array(data_input)

                if len(data_input[0][0])<config.winsize:
                    concat = np.zeros([3, sta_num, config.winsize - len(data_input[0][0])])
                    data_input=np.concatenate([data_input,concat],axis=2)

                if len(data_input[0][0])>config.winsize:
                    data_input=data_input[:, :, :config.winsize]

                data_input=data_input.transpose((1,2,0))
                data_input = np.array([data_input])
                #print(event_list)

            class_pred, confidence = model.classify(sess=sess, input_=data_input)
            confidence0.append(confidence)

            print(class_pred,confidence)
            if class_pred == 1:
                confidence_total.append(confidence)
                start_total.append(windowed_st[0].stats.starttime)
                end_total.append(windowed_st[0].stats.endtime)

                if start_flag == -1:
                    start_flag = windowed_st[0].stats.starttime
                    end_flag = windowed_st[0].stats.endtime
                else:
                    end_flag = windowed_st[0].stats.endtime
            print(class_pred,start_flag,end_flag,windowed_st[0].stats.starttime)

            if class_pred == 0 and start_flag != -1:  #end_flag < windowed_st[0].stats.starttime:

                confidence = np.max(confidence_total)
                for j in range(len(confidence_total)):
                    if confidence == confidence_total[j]:
                        break
                start_local = start_total[j]
                end_local = end_total[j]
                a=True


                # event = [file[0][0].split('/')[-2], start_flag, end_flag,
                #          confidence, start_local, end_local]
                event = [file[0][0].split('/')[-2], start_flag, end_flag, confidence]

                confidence_total=[]
                start_total = []
                end_total = []

                if plot:
                    plot_num = int(plot_num + 1)
                    name = config.root + '/event_detect/detect_result/png/' \
                           + str(plot_num) + '_' + str(confidence) + '.png'
                    plot_traces.plot(starttime=start_flag, endtime=end_flag, size=(800, 800),
                                    automerge=False, equal_scale=False, linewidth=0.8, outfile=name)

                # print(event)

                event_list.append(event)
                #print(event_list)

                start_flag = -1
                end_flag = -1

            if class_pred == 1 and end_flag+config.winlag / samples_trace>=endtime:
                confidence = np.max(confidence_total)
                for j in range(len(confidence_total)):
                    if confidence == confidence_total[j]:
                        break
                start_local = start_total[j]
                end_local = end_total[j]

                if plot:
                    plot_num = int(plot_num + 1)
                    name = config.root + '/event_detect/detect_result/png/' \
                           + str(plot_num) + '_' + str(confidence) + '.png'
                    plot_traces.plot(starttime=start_flag, endtime=endtime, size=(800, 800),
                                     automerge=False, equal_scale=False, linewidth=0.8, outfile=name)

                # event = [file[0][0].split('/')[-2], start_flag, endtime,
                #          confidence, start_total, end_total]
                event = [file[0][0].split('/')[-2], start_flag, endtime, confidence]

                event_list.append(event)
                start_flag = -1
                end_flag = -1

        if len(event_list) != 0:
            with open(config.root + '/event_detect/detect_result/' + model_name + '/events_list.csv', mode='a', newline='') as f:
                csvwriter = csv.writer(f)
                for event in event_list:
                    csvwriter.writerow(event)
                f.close()

        start_point += 1
        with open(config.root + '/event_detect/detect_result/' + model_name + '/checkpoint', mode='w') as f:
            f.write(str(start_point))
            end = datetime.datetime.now()
            print('{} scanned, num {}, time {}.'.format(file[0][0].split('/')[-2], start_point, end - begin))
            print('checkpoint saved.')


if __name__ == '__main__':
    main('cnn', new_scan=True)
