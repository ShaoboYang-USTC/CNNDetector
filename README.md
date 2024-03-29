# CNN Earthquake Detection Program

by Shaobo Yang, University of Science and Technology of China, 2020

E-mail: <yang0123@mail.ustc.edu.cn>

References: Shaobo Yang, Jing Hu, Haijiang Zhang, Guiquan Liu; Simultaneous Earthquake Detection on Multiple Stations via a Convolutional Neural Network. *Seismological Research Letters, 92*(1), 246-260. doi: [https://doi.org/10.1785/0220200137](https://doi.org/10.1785/0220200137).

This repository is used to store scripts and dataset.

## 1. Installation

* Download repository
* Install dependencies: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`
* Download and unzip test data: `after.zip`

* Download and unzip training data (if training is required): `Events.zip`, `Noise.zip`

  We only provide part of our dataset as an example

* Move these three folders to `CNNDetector/data/`

## 2. Training

A well-trained model path: `CNNDetector/event_detect/saver/cnn/`

If you only want to apply our method to a new dataset, you can pass this step and directly try our well-trained model without training.

If the well-trained model performs bad on your dataset or you want to optimize the CNN architecture, you can modify this code: `CNNDetector/cnn.py`. And you have to modify the configuration file: `CNNDetector/config/config.py`

## 3. Applications

* Put your prepared data set into `CNNDetector/data/after`

* Data preprocessing:
  * Convert data format to SAC
  * Prepare one-day continues 3-component (E, N, Z) waveforms from multiple seismic stations and place them in a folder named by the date
  * The waveforms must have the same initial time and end time
  
* Write seismic station information into `CNNDetector/config/group_info`
  
  If there are too many seismic stations, in order to better use our method, we recommend that you devide these stations into multiple groups based on their location, and each group contains up to 20 stations. `CNNDetector/config/group_info` is also the grouping information file and each group start with '#'. You can modify this file to devide all stations into multiple groups.
  
* Check the parameters in the configuration file: `CNNDetector/config/config.py`

* Run the detection program: `python eq_detect.py` 

## 4. Detection results

* An event list: `CNNDetector/event_detect/detect_result/cnn/events_list.csv`

  <img src="./CNNDetector/event_detect/detect_result/cnn/detection_results.jpg" alt="detection_results" style="zoom:28%;" />

* Figures: `CNNDetector/event_detect/detect_result/png/`

  The figure name follows the format of `ID_probability.png`

<img src="./CNNDetector/event_detect/detect_result/waveform.png" alt="waveform" style="zoom:72%;" />

* Cut waveforms: `CNNDetector/event_detect/detect_result/cut/`

