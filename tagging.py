import warnings; warnings.simplefilter('ignore')
import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import zipfile
import librosa
from portable_m2d import PortableM2D
classes = pd.read_csv('class_labels_indices.csv').sort_values('mid').reset_index()
model = PortableM2D(weight_file='m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth', num_classes=527)
from IPython.display import display, Audio

def show_topk(classes, m2d, wav_file, k=5):#classes:包含类别名称的DataFrame m2d:模型
#wav_file:输入音频文件路径 k=5:默认返回概率最高的5个类别
    print(wav_file)
    # Loads and shows an audio clip.
    wav, _ = librosa.load(wav_file, mono=True, sr=m2d.cfg.sample_rate)#使用librosa加载音频文件
#mono=True转换为单声道 sr=m2d.cfg.sample_rate:避免重采样 返回值:wav:numpy数组 _:librosa默认返回(audio,sr)
    display(Audio(wav, rate=m2d.cfg.sample_rate))#播放音频
    wav = torch.tensor(wav).unsqueeze(0)#将numpy数组转为tensor unsqueeze(0):增加batch维度
    # Predicts class probabilities for the batch segments.
    with torch.no_grad():
        probs = m2d(wav).squeeze(0).softmax(0)
#计算音频的分类概率 torch.no_grad()禁用梯度计算 m2d(wav)模型前向传播,输出logits squeeze(0):移除batch维度
#softmax(0):在dim=0计算softmax
    # Shows the top-k prediction results.
    topk_values, topk_indices = probs.topk(k=k)
#使用torch.topk获取概率最高的k个类别及其概率值tensor
    print(', '.join([f'{classes.loc[i].display_name} ({v*100:.1f}%)' for i, v in zip(topk_indices.numpy(), topk_values.numpy())]))
    print()

files = list(Path('2523056').glob('*.ogg'))
files = np.random.choice(files, size=3, replace=False)

for fn in files:
    show_topk(classes, model, fn)

