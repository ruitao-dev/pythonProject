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
files = list(Path('2523056').glob('*.ogg'))
files = np.random.choice(files, size=3, replace=False)
def show_topk_for_all_frames(classes, m2d, wav_file, k=5):
    print(wav_file)
    # Loads and shows an audio clip.
    wav, _ = librosa.load(wav_file, mono=True, sr=m2d.cfg.sample_rate)
    display(Audio(wav, rate=m2d.cfg.sample_rate))
    #加载并播放音频
    wav = torch.tensor(wav)#转换为tensor
    # Predicts class probabilities for all frames.
    with torch.no_grad():
        logits_per_chunk, timestamps = m2d.forward_frames(wav.unsqueeze(0))  # logits_per_chunk: [1, 62, 527], timestamps: [1, 62]
        probs_per_chunk = logits_per_chunk.squeeze(0).softmax(-1)  # logits [1, 62, 527] -> probabilities [62, 527]
        timestamps = timestamps[0] # [1, 62] -> [62]
        #逐帧推理
    # Shows the top-k prediction results.
    for i, (probs, ts) in enumerate(zip(probs_per_chunk, timestamps)):
        topk_values, topk_indices = probs.topk(k=k)
        sec = f'{ts/1000:.1f}s '
        print(sec, ', '.join([f'{classes.loc[i].display_name} ({v*100:.1f}%)' for i, v in zip(topk_indices.numpy(), topk_values.numpy())]))
        #逐帧输出top-k结果
    print()
show_topk_for_all_frames(classes, model, files[0])
