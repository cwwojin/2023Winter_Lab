# Author
# Soohwan Kim, Seyoung Bae, Cheolhwang Won, Soyoung Cho, Jeongwon Kwak

DATASET_PATH="./kspon/KsponSpeech"
TEST_DATASET_PATH="./kspon/KsponSpeech_scripts"
SAVE_DEST='./manifest'
OUTPUT_UNIT='character'                                          # you can set ONLY "character"
PREPROCESS_MODE='phonetic'                                       # phonetic : "칠 십 퍼센트",  spelling : "70%"
VOCAB_SIZE=5000                                                  # if you use subword output unit, set vocab size
SPLIT='train'
TEST_SPLIT='test'
TRAIN_SIZE=620000

echo "Pre-process KsponSpeech Dataset.."

python main.py \
--dataset_path $DATASET_PATH \
--savepath $SAVE_DEST \
--output_unit $OUTPUT_UNIT \
--preprocess_mode $PREPROCESS_MODE \
--vocab_size $VOCAB_SIZE \
--split $SPLIT \

echo "Pre-process KsponSpeech-Eval Dataset.."

python main.py \
--dataset_path $TEST_DATASET_PATH \
--savepath $SAVE_DEST \
--output_unit $OUTPUT_UNIT \
--preprocess_mode $PREPROCESS_MODE \
--vocab_size $VOCAB_SIZE \
--split $TEST_SPLIT \

echo "Train / Dev Split.."

python scripts/train_test_split.py \
--manifest_path $SAVE_DEST \
--train_size $TRAIN_SIZE \