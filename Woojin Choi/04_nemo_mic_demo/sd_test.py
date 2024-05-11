#Mic Test - sounddevice

import os
import argparse
import sounddevice as sd
import soundfile as sf
import torch
import nemo
import nemo.collections.asr as nemo_asr

CHANNELS = 1    #mono
RATE = 16000    #16,000hz samples
CHECKPOINT_DIR = "models"

if torch.cuda.is_available():
    device = "cuda"
else :
    device = "cpu"

def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str,
                        default='ko',
                        )
    parser.add_argument('--frame', type=int,
                        default=5,
                        help="frame-size (N) : record for N seconds each time.",
                        )                      
    return parser

def main() :
    parser = _get_parser()
    opt = parser.parse_args()
    LANG = opt.lang
    RECORD_SECONDS = opt.frame
    if LANG == 'ko' :
        MODEL = "cwwojin/stt_kr_conformer_ctc_small_20"
    elif LANG == 'en' :
        MODEL = "stt_en_conformer_ctc_small"
    else :
        raise ValueError("supported languages : en, ko")

    if not os.path.exists("audio") :
        os.makedirs("audio")
    if not os.path.exists(CHECKPOINT_DIR) :
        os.makedirs(CHECKPOINT_DIR)    
    
    # set sounddevice defaults
    sd.default.samplerate = RATE
    sd.default.channels = CHANNELS

    #load pretrained NeMo model
    restore_path = f"{CHECKPOINT_DIR}/model-{LANG}.nemo"
    if os.path.isfile(restore_path) :
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=restore_path, map_location=device)
    else :
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL, map_location=device)
        asr_model.save_to(restore_path)
    asr_model.eval();

    # MAIN LOOP : get keyboard input to start recording
    audio_path = f"audio/output.wav"
    streaming = True
    inputs = ['y','n']
    while streaming :
        #get keyboard input
        command = None
        while command not in inputs :
            print("[y/n] 'y' to record / 'n' for shutdown.")
            command = str(input())

        if command == 'y' :
            print("Start to record the audio.")
            mydata = sd.rec(int(RATE * RECORD_SECONDS),blocking=True)
            sf.write(audio_path, mydata, RATE)
            print("recording done.")

            # transcribe w/ model
            result = asr_model.transcribe([audio_path])
            for t in result :
                print(f"transcription : ${t}$ ")

            # DO SOMETHING w/ RESULT
            #result

            continue
        elif command == 'n' :
            print("Finished..")
            streaming = False
        else :
            raise ValueError(f"invalid command : {command}")


if __name__ == '__main__':
    main()