import warnings; warnings.simplefilter('ignore')
import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import zipfile
import librosa
classes = pd.read_csv('class_labels_indices.csv').sort_values('mid').reset_index()
from portable_m2d import PortableM2D
model = PortableM2D(weight_file='m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth', num_classes=527)
from IPython.display import display, Audio
files = list(Path('685484').glob('*.ogg'))
#files = np.random.choice(files, size=3, replace=False)
def repeat_if_short(w, min_duration=48000):#处理短音频片段，使其满足模型输入要求
    while w.shape[-1] < min_duration:
        w = np.concatenate([w, w], axis=-1)
    return w[..., :min_duration]

def show_topk_sliding_window(classes, m2d, wav_file, k=5, hop=1, duration=2.):#通过滑动窗口对长音频分段推理，输出每段的top-k结果
    print(wav_file)
    # Loads and shows an audio clip.
    wav, sr = librosa.load(wav_file, mono=True, sr=m2d.cfg.sample_rate)
    display(Audio(wav, rate=sr))#加载音频
    # Makes a batch of short segments of the wav into wavs, cropped by the sliding window of [hop, duration].
    wavs = [wav[int(c * sr) : int((c + duration) * sr)] for c in np.arange(0, wav.shape[-1] / sr, hop)]
    wavs = [repeat_if_short(wav) for wav in wavs]#生成滑动窗口片段
    wavs = torch.tensor(wavs)
    # Predicts class probabilities for the batch segments.
    with torch.no_grad():
        probs_per_chunk = m2d(wavs).softmax(1)#将分段音频转为tensor
    # Shows the top-k prediction results.
    for i, probs in enumerate(probs_per_chunk):
        topk_values, topk_indices = probs.topk(k=k)
        sec = f'{i * hop:d}s '
        print(sec, ', '.join([f'{classes.loc[i].display_name} ({v*100:.1f}%)' for i, v in zip(topk_indices.numpy(), topk_values.numpy())]))
    #输出每段top-k结果
    print()
#fn0="452838/C16F2448000244_L31.3.1_20250715111336_AB000604C_FAR.ogg"
#fn1="452838/C16F2448000244_L31.3.1_20250715111336_AB000604C_NEAR.ogg"
for fn in files:
    show_topk_sliding_window(classes,model,fn)
#show_topk_sliding_window(classes,model,fn1)
