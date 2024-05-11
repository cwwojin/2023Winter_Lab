# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd

import numpy as np
import librosa
import json

import wave
def convert_to_wav(filepath):
    with open(filepath, "rb") as r:
        data = r.read()
        with wave.open(filepath.replace('pcm', 'wav'), "wb") as w:
            w.setparams((1, 2, 16000, 0, "NONE", "NONE"))
            w.writeframes(data)


def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels["freq"]

    for (id_, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        except KeyError:
            continue

    return target[:-1]


def generate_character_labels(transcripts, labels_dest):
    print('create_char_labels started..')

    label_list = list()
    label_freq = list()

    for transcript in transcripts:
        for ch in transcript:
            if ch not in label_list:
                label_list.append(ch)
                label_freq.append(1)
            else:
                label_freq[label_list.index(ch)] += 1

    # sort together Using zip
    label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
    label = {'id': [0, 1, 2], 'char': ['<pad>', '<sos>', '<eos>'], 'freq': [0, 0, 0]}

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label['id'].append(idx + 3)
        label['char'].append(ch)
        label['freq'].append(freq)

    label['id'] = label['id'][:2000]
    label['char'] = label['char'][:2000]
    label['freq'] = label['freq'][:2000]

    label_df = pd.DataFrame(label)
    label_df.to_csv(os.path.join(labels_dest, "aihub_labels.csv"), encoding="utf-8", index=False)


def generate_character_script(audio_paths, transcripts, labels_dest="", split="train"):
    print('create_script started..')
    #char2id, id2char = load_label(os.path.join(labels_dest, "aihub_labels.csv"))

    # with open(os.path.join("transcripts.txt"), "w") as f:
    #     for audio_path, transcript in zip(audio_paths, transcripts):
    #         char_id_transcript = sentence_to_target(transcript, char2id)
    #         audio_path = audio_path.replace('txt', 'pcm')
    #         f.write(f'{audio_path}\t{transcript}\t{char_id_transcript}\n')
    
    #NEW : generate NeMO-manifest .json file
    manifest_filename = "manifest.json" if split=="train" else "eval_manifest.json"

    with open(os.path.join(labels_dest, manifest_filename), "w") as f:
        for audio_path, transcript in zip(audio_paths, transcripts):
            audio_path = audio_path.replace('txt', 'pcm')
            
            if split == "train":
                audio_path = os.path.join("./kspon/KsponSpeech", audio_path)
            else :
                audio_path = os.path.join("./kspon", audio_path)

            #pcm > wav conversion
            if not os.path.isfile(audio_path.replace('pcm', 'wav')) :
                convert_to_wav(audio_path)
            audio_path = audio_path.replace('pcm', 'wav')
            (audio, _) = librosa.load(audio_path, sr=16000)
            duration = librosa.core.get_duration(y=audio, sr=16000)

            # duration = len(np.memmap(audio_path, dtype='h', mode='r'))
            # duration = duration / 16000  #sample rate = 16000
            
            # Write the metadata to the manifest
            metadata = {
                "audio_filepath": audio_path,
                "duration": duration,
                "text": transcript
            }
            json.dump(metadata, f, ensure_ascii=True)  #ensure_ascii=True
            f.write('\n')

