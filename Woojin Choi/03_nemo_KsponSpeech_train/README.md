# 1. KsponSpeech-preprocess
* Original Code from https://github.com/sooftware/ksponspeech
* KsponSpeech Dataset : https://aihub.or.kr/ (Dataset name : 한국어 음성)

## Pre-processing KsponSpeech corpus provided by AI Hub
   
Preprocessing KsponSpeech Dataset for Nvidia NeMo models.   
This script produces 3 manifest files for NeMo (train, dev, eval-sets)  
The original .pcm-files (raw) are converted to .wav-files for compatibility.    
- Train : 620000
- Eval : 6000
- Dev/Validation : 2545

### File Structure

```shell
"./kspon/"
|-"KsponSpeech/"
|   |-"KsponSpeech_01/"
|   |..
|   |-"KsponSpeech_05/"
|-"KsponSpeech_eval/"
|-"KsponSpeech_scripts/"
``` 
   
## Prerequisites
* Pandas: `pip install pandas` (Refer [here](https://github.com/pandas-dev/pandas) for problem installing Pandas)  
* Sentencepiece: `pip install sentencepiece` (Refer [here](https://github.com/google/sentencepiece) for problem installing Sentencepiece) 

## Usage

* Run Script (edit configs in .sh file)
```shell
$ ./preprocess.sh
``` 
### Preprocess Configs
1. Raw data
```
b/ (70%)/(칠 십 퍼센트) 확률이라니 아/ (뭐+ 뭔)/(모+ 몬) 소리야 진짜 (100%)(백 프로)가 왜 안돼? n/
``` 
2. Delete noise labels, such as b/, n/, / ..
```
(70%)/(칠 십 퍼센트) 확률이라니 아/ (뭐+ 뭔)/(모+ 몬) 소리야 진짜 (100%)(백 프로)가 왜 안돼?
```
3. Delete labels such as '/', '*', '+', etc. (used for interjection representation)
```
(70%)/(칠 십 퍼센트) 확률이라니 아 (뭐 뭔)/(모 몬) 소리야 진짜 (100%)(백 프로)가 왜 안돼?
```
#### Choose between phonetic transcription / spelling transcription. (edit "preprocess.sh")
* (1) : phonetic transcript
```
칠 십 퍼센트 확률이라니 아 모 몬 소리야 진짜 백 프로가 왜 안돼?
```
* (2) : spelling transcript
```
70% 확률이라니 아 뭐 뭔 소리야 진짜 100%가 왜 안돼?
```
#### The Script takes a while.

# 2. NeMo : Korean ASR w/ KsponSpeech
## Building Tokenizer
* Run Script (edit configs in .sh file)
```shell
$ ./tokenizer.sh
``` 
## Training Model
* CTC-models supported. (Transducer-models NOT supported)
* Korean w/ KsponSpeech ONLY
### Usage
```shell
$ mkdir models
$ python asr_train_ko.py \
    --vocab_size=5000 \
    --pretrained_model=stt_en_conformer_ctc_small \
    --batch_size=32 \
    --epochs=20 \
    --num_workers=16 \
    --devices=1
``` 
- vocab_size : should be the same as the tokenizer made from running "tokenizer.sh"
- pretrained_model : model name from Huggingface / Nvidia
- devices : number of GPU's. For multi-GPU training, edit Trainer configs.
### Logging
- W&B logger is available for Nvidia NeMo
- To use, un-comment and edit "ExpManagerConfig" part
# Author
- Woojin Choi / cwwojin@kaist.ac.kr