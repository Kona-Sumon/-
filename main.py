import h5py
import numpy as np

# Path to the dataset
dataset_path = './indy_20160630_01.mat'

# Open the MATLAB file using h5py
mat_file = h5py.File(dataset_path)

# Display the keys in the MATLAB file
for key in mat_file.keys():
    print(mat_file[key])

# wf_refs
waveforms_refs = []

# Iterate through the 'wf' dataset in the MATLAB file
for waveform_cell in mat_file['wf']:
    # Convert the waveform cell to a NumPy array and append to the list
    waveform_array = np.array(waveform_cell)
    waveforms_refs.append(waveform_array)

# Convert the list of waveforms to a NumPy array and transpose it
waveforms_refs = np.array(waveforms_refs).T

# spikes_refs
spikes_refs = []

# Iterate through the 'spikes' dataset in the MATLAB file
for spikes_cell in mat_file['spikes']:
    # Convert the spikes cell to a NumPy array and append to the list
    spikes_array = np.array(spikes_cell)
    spikes_refs.append(spikes_array)

# Convert the list of spikes to a NumPy array and transpose it
spikes_refs = np.array(spikes_refs).T

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def channel_cluster_num(refs) -> int:
    """ 
    返回该通道被分为几类
        [1:4]：类
        0: 表示只有一列2维数据
        -1: 表示没有2维数据
    """
    type_num = -1
    for ref in refs:
        data = mat_file[ref]
        if len(data.shape) > 1:
            type_num += 1
    return type_num

from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=600, highcut=3000, fs=30000, order=5):
    """
    带通滤波器

    参数：
    data：输入信号
    lowcut：低截止频率（单位：赫兹）
    highcut：高截止频率（单位：赫兹）
    fs：采样频率（单位：赫兹）
    order：滤波器阶数

    返回值：
    滤波后的信号
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    y = filtfilt(b, a, data)
    return y

def normalize_amplitude(spikes):
    """
    对spikes做幅度标准化
    
    Parameters:
    spikes (numpy.ndarray): spikes波形的 (N, 48) 数组

    Returns:
    返回幅度标准化后的spikes数组，维度还是(N, 48)不变
    """
    # Find the maximum amplitude of each waveform
    max_amplitudes = np.max(np.abs(spikes), axis=1, keepdims=True)
    
    normalized_spikes = spikes / max_amplitudes
    return normalized_spikes

from scipy.signal import find_peaks

def extract_wave_features(spikes, width_thresh=None, height_thresh=None):
    """
    提取波形的特征：波峰和波谷的发生时间以及持续长度。

    Parameters:
    spikes (numpy.ndarray): (N, 48) 的波形数组。
    width_thresh (float, optional): 波峰/波谷宽度的阈值。
    height_thresh (float, optional): 波峰/波谷高度的阈值。

    Returns:
    numpy.ndarray: 波峰和波谷的特征数组，形状为 (N, 4)。
    每行包含：第一个波峰的位置、第一个波峰的宽度、第一个波谷的位置、第一个波谷的宽度。
    """
    N, _ = spikes.shape
    features = np.zeros((N, 4))  # 初始化特征数组

    for i in range(N):
        spike = spikes[i, :]

        # 查找波峰
        peaks, _ = find_peaks(spike, width=width_thresh, height=height_thresh)
        if len(peaks) > 0:
            features[i, 0] = peaks[0]  # 第一个波峰的位置
            # 计算波峰宽度
            if width_thresh is not None:
                features[i, 1] = _['widths'][0]  # 第一个波峰的宽度

        # 查找波谷（在反转的信号中查找波峰）
        valleys, _ = find_peaks(-spike, width=width_thresh, height=height_thresh)
        if len(valleys) > 0:
            features[i, 2] = valleys[0]  # 第一个波谷的位置
            # 计算波谷宽度
            if width_thresh is not None:
                features[i, 3] = _['widths'][0]  # 第一个波谷的宽度

    return features

import scipy.stats as stats

def calculate_energy_parts(spikes):
    """ 计算正负部分的能量和 """
    # 分离正负部分
    positive_parts = np.maximum(spikes, 0)
    negative_parts = np.minimum(spikes, 0)

    # 计算能量和
    positive_energy = np.sum(positive_parts**2, axis=1)
    negative_energy = np.sum(negative_parts**2, axis=1)

    return positive_energy, negative_energy

def cal_self_correlation(spikes, max_lag=12):
    """
    计算 (N, 48) 形状的波形数组中每个波形的自相关系数。

    Parameters:
    spikes (numpy.ndarray): (N, 48) 的波形数组。
    max_lag (int): 要计算的最大延迟。

    Returns:
    numpy.ndarray: 自相关系数数组，形状为 (N, max_lag)。
    """
    N, _ = spikes.shape
    autocorr_results = np.zeros((N, max_lag))

    for i in range(N):
        series = spikes[i, :]
        n = len(series)
        mean = np.mean(series)
        autocorr = np.correlate(series - mean, series - mean, mode='full') / np.var(series) / n
        autocorr_results[i, :] = autocorr[n-1:n-1+max_lag]

    return autocorr_results

def feature_extraction(spikes):
    """
    提取波形的特征。

    Parameters:
    spikes (numpy.ndarray): (N, 48) 的波形数组。

    Returns:
    numpy.ndarray: 特征数组，形状为 (N, ...)，具体特征内容见实现部分。
    """
    # 信号振幅（正峰值和负峰值）
    peak_amplitudes_plus = np.max(spikes, axis=1)
    peak_amplitudes_minus = np.abs(np.min(spikes, axis=1))
    
    # 信号能量：信号的平方和（正负两部分）
    positive_energy, negative_energy = calculate_energy_parts(spikes)
    
    # 偏斜度（Skewness）
    skewness = stats.skew(spikes, axis=1)
    
    # 峰度（Kurtosis）
    kurtosis = stats.kurtosis(spikes, axis=1)
    
    # 均值、标准差
    mean_values = np.mean(spikes, axis=1)
    std_devs = np.std(spikes, axis=1)
    
    # 自相关系数
    self_corr = cal_self_correlation(spikes, 16)
    
    # 波峰波谷
    peak_features = extract_wave_features(spikes, width_thresh=2, height_thresh=0.8)
    
    # TODO：小波系数的统计特征：小波变换可以分解波形的不同频率组成，提供更多关于波形的信息
    # TODO：自相关特征：衡量波形与其自身在不同时间滞后下的相似度。
    # TODO：波峰和波谷的数量：反映波形的复杂度。
    # TODO：波峰和波谷的均值和标准差：分别描述正负极值的统计特性。
    
    return np.column_stack((
                            peak_amplitudes_plus,       
                            peak_amplitudes_minus, 
                            # positive_energy, 
                            # negative_energy, 
                            # skewness, 
                            # kurtosis, 
                            # mean_values, 
                            # std_devs,
                            # self_corr,
                            peak_features,
    ))

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def channel_classification(channel):
    """
    对某一通道进行分类。

    Parameters:
    channel (int): 通道索引。

    Returns:
    None
    """
    # 读取数据
    refs = waveforms_refs[channel]
    n_clusters = channel_cluster_num(refs)
    if n_clusters < 2:
        print(f'[ERROR] clusters_num = {n_clusters}')
        return
    unsorted_data = np.array(mat_file[refs[0]]).T   # (42158, 48)
    
    # 滤波
    filtered_data = bandpass_filter(unsorted_data)
    
    # 幅度标准化
    std_filtered_data = normalize_amplitude(filtered_data)
    
    # 特征提取
    features = feature_extraction(std_filtered_data)
    
    # 特征标准化
    scaler = StandardScaler()
    std_features = scaler.fit_transform(features)
    print(f'std_features.shape = {std_features.shape}')
    
    # PCA降维
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(std_features)
    print(f'data_reduced.shape = {data_reduced.shape}')
    
    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data_reduced)
    
    # 获取聚类标签和聚类中心
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # 绘制结果
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'y']
    
    for i in range(kmeans.n_clusters):
        # 绘制聚类数据点
        plt.scatter(data_reduced[labels == i, 0], data_reduced[labels == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
    
    # 绘制聚类中心
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='yellow', marker='*', edgecolor='black', label='Centroids')
    plt.title(f"K-Means Clustering on PCA-Reduced Data (channel={channel})")
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.legend()
    plt.show()


# 效果较好的通道：22, 54, 74， 78, 92
for i in range(96):
    channel_classification(i)


def one_channel_classification(channel, single_data_index=0):
    """
    单通道分类。

    Parameters:
    channel (int): 通道索引。
    single_data_index (int): 单个数据索引。

    Returns:
    None
    """
    refs = waveforms_refs[channel]
    type_nums = channel_cluster_num(refs)
    if type_nums < 2:
        print('[WARNING] channel\'s type < 2')
        return
    data = np.array(mat_file[refs[0]]).T    # 0: 第一列的unsorted_data
    
    x = range(data.shape[-1])
    
    # 原始数据
    plt.figure(figsize=(2, 1.5))
    plt.title('Single Raw Spike')
    plt.plot(x, data[single_data_index])
    plt.xlabel(f'Time: {len(x)}ms after each spike event (ms)')
    plt.ylabel('Voltage (uV)')
    plt.savefig('./img/single_raw_spike.png')
    
    # 带通滤波
    lowcut = 600
    highcut = 3000
    fs = 30000
    order = 5
    
    filtered_data = np.array([bandpass_filter(y_m, lowcut, highcut, fs, order) for y_m in data])
    
    # 单个滤波spike
    single_filtered_spike = bandpass_filter(data[single_data_index], lowcut, highcut, fs, order)
    std_single_filtered_spike = normalize_amplitude(single_filtered_spike.reshape(1, -1))
    plt.figure(figsize=(2, 1.5))
    plt.title(f'Single Filtered Spikes ({lowcut}~{highcut}Hz)')
    plt.plot(x, std_single_filtered_spike.reshape(-1))
    plt.xlabel(f'Time: {len(x)}ms after each spike event (ms)')
    plt.ylabel('Voltage (uV)')
    plt.savefig('./img/single_filtered_spike.png')
    
    # Spike锋电位检测
    abs_max = np.max(np.abs(filtered_data))
    std_filtered_data = single_filtered_spike / abs_max
    
    plt.figure(figsize=(2, 1.5))
    plt.title(f'Std Single Filtered Spikes ({lowcut}~{highcut}Hz)')
    plt.plot(x, std_filtered_data)
    plt.xlabel(f'Time: {len(x)}ms after each spike event (ms)')
    plt.ylabel('Voltage (uV)')
    plt.savefig('./img/std_single_filtered_spike.png')
    
    # 多个滤波spikes
    plt.figure()
    plt.title(f'Multiple Filtered Spikes ({lowcut}~{highcut}Hz)')
    for y_m in filtered_data:
        plt.plot(x, y_m, linewidth=0.5, color='blue')
    plt.xlabel(f'Time: {len(x)}ms after each spike event (ms)')
    plt.ylabel('Voltage (uV)')
    plt.savefig('./img/multiple_filtered_spikes.png')

one_channel_classification(channel=1, single_data_index=2)
for i in range(20):
    one_channel_classification(channel=1, single_data_index=1000 + i)

