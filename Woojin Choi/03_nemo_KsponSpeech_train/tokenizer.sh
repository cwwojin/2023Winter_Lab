#build tokenizer - Korean

TOKENIZER_TYPE="bpe" #@param ["bpe", "unigram"]
VOCAB_SIZE=5000
TRAIN_MANIFEST="manifest/train_manifest.json" 
TOKENIZER_DIR="tokenizers"
COVERAGE=1

echo "Build Tokenizer.."

python scripts/process_asr_text_tokenizer.py \
  --manifest=$TRAIN_MANIFEST \
  --vocab_size=$VOCAB_SIZE \
  --data_root=$TOKENIZER_DIR \
  --tokenizer="spe" \
  --spe_type=$TOKENIZER_TYPE \
  --spe_character_coverage=$COVERAGE \
  --no_lower_case \
  --log