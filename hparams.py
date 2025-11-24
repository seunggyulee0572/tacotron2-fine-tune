# import tensorflow as tf
# from text import symbols


# def create_hparams(hparams_string=None, verbose=False):
#     """Create model hyperparameters. Parse nondefault from given string."""

#     hparams = tf.contrib.training.HParams(
#         ################################
#         # Experiment Parameters        #
#         ################################
#         epochs=500,
#         iters_per_checkpoint=1000,
#         seed=1234,
#         dynamic_loss_scaling=True,
#         fp16_run=False,
#         distributed_run=False,
#         dist_backend="nccl",
#         dist_url="tcp://localhost:54321",
#         cudnn_enabled=True,
#         cudnn_benchmark=False,
#         ignore_layers=['embedding.weight'],

#         ################################
#         # Data Parameters             #
#         ################################
#         load_mel_from_disk=False,
#         training_files='filelists/kr_audio_text_train_filelist.txt',
#         validation_files='filelists/kr_audio_text_val_filelist.txt',
#         text_cleaners=['english_cleaners'],

#         ################################
#         # Audio Parameters             #
#         ################################
#         max_wav_value=32768.0,
#         sampling_rate=22050,
#         filter_length=1024,
#         hop_length=256,
#         win_length=1024,
#         n_mel_channels=80,
#         mel_fmin=0.0,
#         mel_fmax=8000.0,

#         ################################
#         # Model Parameters             #
#         ################################
#         n_symbols=len(symbols),
#         symbols_embedding_dim=512,

#         # Encoder parameters
#         encoder_kernel_size=5,
#         encoder_n_convolutions=3,
#         encoder_embedding_dim=512,

#         # Decoder parameters
#         n_frames_per_step=1,  # currently only 1 is supported
#         decoder_rnn_dim=1024,
#         prenet_dim=256,
#         max_decoder_steps=1000,
#         gate_threshold=0.5,
#         p_attention_dropout=0.1,
#         p_decoder_dropout=0.1,

#         # Attention parameters
#         attention_rnn_dim=1024,
#         attention_dim=128,

#         # Location Layer parameters
#         attention_location_n_filters=32,
#         attention_location_kernel_size=31,

#         # Mel-post processing network parameters
#         postnet_embedding_dim=512,
#         postnet_kernel_size=5,
#         postnet_n_convolutions=5,

#         ################################
#         # Optimization Hyperparameters #
#         ################################
#         use_saved_learning_rate=False,
#         learning_rate=1e-3,
#         weight_decay=1e-6,
#         grad_clip_thresh=1.0,
#         batch_size=64,
#         mask_padding=True  # set model's padded outputs to padded values
#     )

#     if hparams_string:
#         tf.logging.info('Parsing command line hparams: %s', hparams_string)
#         hparams.parse(hparams_string)

#     if verbose:
#         tf.logging.info('Final parsed hparams: %s', hparams.values())

#     return hparams
# TensorFlow 제거하고 argparse 사용
import argparse
from text.symbols import (
    symbols_english,
    symbols_korean_jamo,
    symbols_korean_romanized,
)

SYMBOL_SET = 'korean_romanized'

_symbol_sets = {
    'english': symbols_english,
    'korean_jamo': symbols_korean_jamo,
    'korean_romanized': symbols_korean_romanized,
}

def get_symbols(symbol_set: str):
    if symbol_set not in _symbol_sets:
        raise ValueError(f"Unknown SYMBOL_SET={symbol_set}. "
                         f"Choose one of {list(_symbol_sets.keys())}")
    return _symbol_sets[symbol_set]

class HParams:
    """간단한 hyperparameter 저장 클래스"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def parse(self, hparams_string):
        """커맨드라인 파라미터 파싱"""
        if hparams_string:
            pairs = hparams_string.split(',')
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=')
                    key = key.strip()
                    value = value.strip()
                    
                    # 타입 변환
                    if hasattr(self, key):
                        original_value = getattr(self, key)
                        if isinstance(original_value, bool):
                            value = value.lower() == 'true'
                        elif isinstance(original_value, int):
                            value = int(value)
                        elif isinstance(original_value, float):
                            value = float(value)
                        elif isinstance(original_value, list):
                            value = value.strip('[]').split(',')
                    
                    setattr(self, key, value)
    
    def values(self):
        """모든 hyperparameter 딕셔너리로 반환"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""
    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=500,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=True,
        distributed_run=True,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/train.txt',
        validation_files='filelists/val.txt',
        text_cleaners=['korean_cleaners_jamo'],
        symbol_set='korean_jamo',

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(get_symbols('korean_jamo')),
        # n_symbols=len(get_symbols(hparams.symbol_set)),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=64,
        mask_padding=True  # set model's padded outputs to padded values
    )
    
    # hparams.add_hparam('n_symbols', len(get_symbols(hparams.symbol_set)))

    if hparams_string:
        print('Parsing command line hparams: %s' % hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        print('Final parsed hparams: %s' % hparams.values())

    return hparams
