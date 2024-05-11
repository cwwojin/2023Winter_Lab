#py_script version
#'ko' : pre-trained Eng model -> KsponSpeech

import os
import argparse
import copy
from omegaconf import OmegaConf, open_dict
import pytorch_lightning as ptl
import wandb

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import exp_manager

def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int,
                        default=5000,
                        )
    parser.add_argument('--pretrained_model', type=str,
                        default="stt_en_conformer_ctc_small",
                        )
    parser.add_argument('--batch_size', type=int,
                        default=32,
                        )
    parser.add_argument('--epochs', type=int,
                        default=20,
                        )
    parser.add_argument('--num_workers', type=int,
                        default=16,
                        )
    parser.add_argument('--devices', type=int,
                        default=1,
                        )
    return parser


LANGUAGE = "ko"
TOKENIZER_TYPE = "bpe" #@param ["bpe", "unigram"]

def main() :
    parser = _get_parser()
    opt = parser.parse_args()
    VOCAB_SIZE = opt.vocab_size
    TOKENIZER_DIR = f"tokenizers/tokenizer_spe_{TOKENIZER_TYPE}_v{VOCAB_SIZE}/"
    PRETRAINED_MODEL = opt.pretrained_model    
    NUM_WORKERS = opt.num_workers
    EPOCHS = opt.epochs
    BATCH_SIZE = opt.batch_size
    DEVICES = opt.devices

    #Train one ASR model (Korean, KsponSpeech, fine-tune from pretrained)
    train_manifest = "manifest/train_manifest.json"
    dev_manifest = "manifest/dev_manifest.json"
    test_manifest = "manifest/eval_manifest.json"

    #Load pretrained model
    save_path = f"models/Model-{LANGUAGE}_{PRETRAINED_MODEL}_{EPOCHS}.nemo"
    if os.path.isfile(save_path) :
        model = nemo_asr.models.ASRModel.restore_from(restore_path=save_path, map_location='cpu')
    else :
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=PRETRAINED_MODEL, map_location='cpu')
    model.change_vocabulary(new_tokenizer_dir=TOKENIZER_DIR, new_tokenizer_type="bpe")
    cfg = copy.deepcopy(model.cfg)
    
    # Setup new tokenizer
    cfg.tokenizer.dir = TOKENIZER_DIR
    cfg.tokenizer.type = TOKENIZER_TYPE
    model.cfg.tokenizer = cfg.tokenizer

    # Setup train, validation, test configs
    with open_dict(cfg):
        # Train dataset
        cfg.train_ds.manifest_filepath = train_manifest
        cfg.train_ds.batch_size = BATCH_SIZE
        cfg.train_ds.num_workers = NUM_WORKERS
        cfg.train_ds.pin_memory = True
        cfg.train_ds.use_start_end_token = True
        cfg.train_ds.is_tarred = False
        #cfg.train_ds.trim_silence = True

        # Validation dataset
        cfg.validation_ds.manifest_filepath = dev_manifest
        cfg.validation_ds.batch_size = BATCH_SIZE
        cfg.validation_ds.num_workers = NUM_WORKERS
        cfg.validation_ds.pin_memory = True
        cfg.validation_ds.use_start_end_token = True
        cfg.validation_ds.is_tarred = False
        #cfg.validation_ds.trim_silence = True

        # Test dataset
        cfg.test_ds.manifest_filepath = test_manifest
        cfg.test_ds.batch_size = BATCH_SIZE
        cfg.test_ds.num_workers = NUM_WORKERS
        cfg.test_ds.pin_memory = True
        cfg.test_ds.use_start_end_token = True
        cfg.test_ds.is_tarred = False
        #cfg.test_ds.trim_silence = True

    # setup model with new configs
    model.setup_training_data(cfg.train_ds)
    model.setup_multiple_validation_data(cfg.validation_ds)
    model.setup_multiple_test_data(cfg.test_ds)

    with open_dict(model.cfg.optim):
        model.cfg.optim.lr = 1.0

    model._wer.use_cer = True
    model._wer.log_prediction = True

    # setup trainer
    trainer = ptl.Trainer(
                        devices=DEVICES, 
                        accelerator='gpu', 
                        # strategy='ddp_find_unused_parameters_false',
                        # sync_batchnorm=True,
                        # precision=16,   #use AMP
                        max_epochs=EPOCHS, 
                        accumulate_grad_batches=1,
                        enable_checkpointing=False,
                        logger=False,
                        log_every_n_steps=1000,
                        check_val_every_n_epoch=1,
                        #detect_anomaly=True,
                        )

    # Setup model with the trainer
    model.set_trainer(trainer)
    model.cfg = model._cfg

    # setup exp-manager + W&B logger
    config = exp_manager.ExpManagerConfig(
        exp_dir=f'experiments/lang-{LANGUAGE}/',
        name=f"ASR-Model-Language-{LANGUAGE}",
        checkpoint_callback_params=exp_manager.CallbackParams(
            monitor="val_wer",
            mode="min",
            always_save_nemo=True,
            save_best_model=True,
        ),
        # create_wandb_logger=True,
        # wandb_logger_kwargs={
        #     'name' : f"{PRETRAINED_MODEL}_{EPOCHS}",
        #     'project' : 'nemo_test_1',
        # }
    )

    config = OmegaConf.structured(config)
    logdir = exp_manager.exp_manager(trainer, config)

    # Train
    trainer.fit(model)

    # Evaluation
    if model.prepare_test(trainer) :
        trainer.test(model)

    # Save model
    model.save_to(f"{save_path}")
    print(f"Model saved at path : {os.getcwd() + os.path.sep + save_path}")

    wandb.finish()


if __name__ == '__main__':
    main()
