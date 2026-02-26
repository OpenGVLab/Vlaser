import logging
from typing import ClassVar

import numpy as np
from scipy.fft import dct
from scipy.fft import idct
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from transformers.processing_utils import ProcessorMixin
import pywt

class UniversalActionProcessor(ProcessorMixin):
    attributes: ClassVar[list[str]] = ["bpe_tokenizer"]
    bpe_tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        bpe_tokenizer: PreTrainedTokenizerFast,
        low_scale: float = 10,
        high_scale: float = 100,
        vocab_size: int = 2048,
        low_min_token: int = 0,
        low_max_token: int = 0,
        high_min_token: int = 0,
        *,
        action_dim: int | None = None,
        time_horizon: int | None = None,
        high_compressed_size: int | None = None,  # time_horizon中高频部分的size: high_compressed_size
        low_compressed_size: int | None = None,  # time_horizon中低频部分的size: low_compressed_size
    ):
        self.low_scale = low_scale
        self.high_scale = high_scale
        self.vocab_size = vocab_size
        self.low_min_token = low_min_token
        self.low_max_token = low_max_token
        self.high_min_token = high_min_token

        # Action horizon and dimension needed during decoding. These can be specified
        # in three ways (in order of priority):
        # 1. passed in as kwargs to decode()
        # 2. in the constructor
        # 3. cached from the last time decode() was called
        self.time_horizon = time_horizon
        self.action_dim = action_dim
        self.called_time_horizon = time_horizon
        self.called_action_dim = action_dim

        self.high_compressed_size = high_compressed_size
        self.low_compressed_size = low_compressed_size


        super().__init__(bpe_tokenizer)

    def __call__(self, action_chunk: np.array) -> np.array: # TODO
        assert action_chunk.ndim <= 3, "Only 3 dimensions supported: [batch, timesteps, action_dim]"
        if action_chunk.ndim == 2:
            action_chunk = action_chunk[None, ...]

        # Cache the time horizon and action dimension for decoding
        self.called_time_horizon = action_chunk.shape[-2]
        self.called_action_dim = action_chunk.shape[-1]

        low_freqs, high_freqs = [], []
        for a in action_chunk:
            low_freq, high_freq = pywt.wavedec(a, 'db2', level=1, axis=0)
            low_freqs.append(low_freq.flatten())
            high_freqs.append(high_freq.flatten())

        low_freqs = np.array(low_freqs) * self.low_scale - self.low_min_token  # 让low_freqs的范围为[0, low_max_token-low_min_token]
        high_freqs = np.array(high_freqs) * self.high_scale + (
                    self.low_max_token - self.low_min_token) - self.high_min_token+1 # 让high_freqs的范围为[low_max_token- low_min_token+1, low_max_token- low_min_token + high_max_token-high_min_token]

        tokens = []
        freqs = np.concatenate((low_freqs, high_freqs), axis=1)
        for token in freqs:
            rounded_tokens = np.around(token)
            rounded_tokens = rounded_tokens.astype(int)
            string = "".join(map(chr, rounded_tokens))
            tokens.append(self.bpe_tokenizer(string)["input_ids"])
        return tokens


    def decode(  # TODO
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.array:
        self.time_horizon = time_horizon or self.time_horizon or self.called_time_horizon
        self.action_dim = action_dim or self.action_dim or self.called_action_dim

        # Cache the time horizon and action dimension for the next call
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim

        assert (
            self.time_horizon is not None and self.action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."

        decoded_actions = []
        for token in tokens:
            try:
                decoded_tokens = self.bpe_tokenizer.decode(token)
                decoded_tokens = np.array(list(map(ord, decoded_tokens)))
                assert len(decoded_tokens) == self.action_dim * (self.high_compressed_size+self.low_compressed_size)

                # 分离低频和高频部分
                low_decoded_tokens = decoded_tokens[:self.action_dim * self.low_compressed_size]
                high_decoded_tokens = decoded_tokens[self.action_dim * self.low_compressed_size:]

                low_decoded_dwt_coeff = low_decoded_tokens + self.low_min_token
                high_decoded_dwt_coeff = high_decoded_tokens + self.high_min_token - (self.low_max_token - self.low_min_token) -1

                low_decoded_dwt_coeff = low_decoded_dwt_coeff.reshape(-1, self.action_dim) / self.low_scale
                high_decoded_dwt_coeff = high_decoded_dwt_coeff.reshape(-1, self.action_dim) / self.high_scale

                assert (
                    low_decoded_dwt_coeff.shape
                    == (
                        self.low_compressed_size,
                        self.action_dim,
                    )
                ), f"Decoded DWT coefficients have shape {low_decoded_dwt_coeff.shape}, expected ({self.time_horizon}, {self.low_compressed_size})"

                assert (
                        high_decoded_dwt_coeff.shape
                        == (
                            self.high_compressed_size,
                            self.action_dim,
                        )
                ), f"Decoded DWT coefficients have shape {high_decoded_dwt_coeff.shape}, expected ({self.time_horizon}, {self.high_compressed_size})"

            except Exception as e:
                print(f"Error decoding tokens: {e}")
                print(f"Tokens: {token}")
                low_decoded_dwt_coeff = np.zeros((self.low_compressed_size, self.action_dim))
                high_decoded_dwt_coeff = np.zeros((self.high_compressed_size, self.action_dim))
            decoded_action = pywt.waverec((low_decoded_dwt_coeff, high_decoded_dwt_coeff), 'db2', axis=0)
            decoded_actions.append(decoded_action)
        return np.stack(decoded_actions)

    @classmethod
    def fit(
        cls,
        action_data: list[np.array],
        low_scale: float = 10,
        high_scale: float = 100,
        vocab_size: int = 2048,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> "UniversalActionProcessor":
        # Run DCT over all inputs

        low_freqs = []
        high_freqs = []

        for a in action_data:
            low_freq, high_freq = pywt.wavedec(a, 'db2', level=1, axis=0)
            low_freqs.append(low_freq.flatten())
            high_freqs.append(high_freq.flatten())
        
        action_dim = action_dim or action_data[0].shape[-1]
        high_compressed_size = high_freq.shape[0]
        low_compressed_size = low_freq.shape[0]
        time_horizon = time_horizon or action_data[0].shape[-2]

        low_max_token = int(np.around(np.concatenate(low_freqs) * low_scale).max())
        low_min_token = int(np.around(np.concatenate(low_freqs) * low_scale).min())

        high_max_token = int(np.around(np.concatenate(high_freqs) * high_scale).max())
        high_min_token = int(np.around(np.concatenate(high_freqs) * high_scale).min())

        low_freqs = np.array(low_freqs)*low_scale - low_min_token  # 让low_freqs的范围为[0, low_max_token-low_min_token]
        high_freqs = np.array(high_freqs)*high_scale+(low_max_token- low_min_token)-high_min_token+1 # 让high_freqs的范围为[low_max_token- low_min_token+1, low_max_token- low_min_token + high_max_token-high_min_token+1]

        low_max, low_min = np.max(low_freqs), np.min(low_freqs)
        high_max, high_min = np.max(high_freqs), np.min(high_freqs)


        low_min_vocab_size = low_max_token - low_min_token + 1
        high_min_vocab_size = high_max_token - high_min_token + 1
        assert (
                low_min_vocab_size+high_min_vocab_size <= vocab_size
        ), f"Vocab size {vocab_size} is too small for the range of tokens {low_min_vocab_size+high_min_vocab_size}"
        if low_min_vocab_size+high_min_vocab_size + 100 > vocab_size:
            logging.warning(
                f"Initial alphabet size {low_min_vocab_size+high_min_vocab_size} is almost as large as the vocab"
                f"size {vocab_size}, consider increasing vocab size"
            )


        # Make token iterator for BPE training
        def _token_iter():
            freqs = np.concatenate((low_freqs, high_freqs), axis=1)
            for token in freqs:
                rounded_tokens = np.around(token)
                rounded_tokens = rounded_tokens.astype(int)
                string = "".join(map(chr, rounded_tokens))
                yield string

        # freqs = np.concatenate((low_freqs, high_freqs), axis=0)
        # for token in freqs:
        #     rounded_tokens = np.around(token)
        #     rounded_tokens = rounded_tokens.astype(int)
        #     string = "".join(map(chr, rounded_tokens))



        # Train BPE tokenizer
        bpe = ByteLevelBPETokenizer()

        # Set up the entire range of possible tokens as the initial alphabet
        alphabet = [chr(i) for i in range(low_min_vocab_size + high_min_vocab_size+1)]
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=[],
            initial_alphabet=alphabet,
            max_token_length=10000,
        )

        # Train the inner tokenizer (don't use ByteLevelBPETokenizer.train_from_iterator()
        # because it doesn't support custom alphabets)
        bpe._tokenizer.train_from_iterator(_token_iter(), trainer=trainer)

        return cls(
            PreTrainedTokenizerFast(tokenizer_object=bpe, clean_up_tokenization_spaces=False),
            low_scale=low_scale,
            high_scale=high_scale,
            vocab_size=vocab_size,
            low_min_token=low_min_token,
            low_max_token=low_max_token,
            high_min_token=high_min_token,

            action_dim= action_dim,
            time_horizon= None,
            high_compressed_size= high_compressed_size,
            low_compressed_size= low_compressed_size
        )
