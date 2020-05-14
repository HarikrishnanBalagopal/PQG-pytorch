import torch
import numpy as np

from spacy.lang.en import English
from torch.utils.data import Dataset, DataLoader
from utils import CUB_200_2011_METADATA_PATH, CUB_200_2011_IMG_ID_TO_CAPS_PATH, WORD_ID_TO_WORD_5K_PATH, WORD_ID_TO_WORD_10K_PATH, QUORA_PARAPHRASE_DATA_PATH

class CUB_200_2011_Paraphrase(Dataset):
    """If should_pad is True, need to also provide a pad_to_length. Padding also adds <START> and <END> tokens to captions."""
    def __init__(self, split='all', should_pad=False, pad_to_length=None, no_start_end=False, **kwargs):
        super().__init__()

        assert split in ('all', 'train_val', 'test')
        if should_pad:
            assert pad_to_length >= 3 # <START> foo <END> need at least length 3.

        self.split = split
        self.should_pad = should_pad
        self.no_start_end = no_start_end
        self.pad_to_length = pad_to_length

        metadata = torch.load(CUB_200_2011_METADATA_PATH)

        # labels
        self.img_id_to_class_id = metadata['img_id_to_class_id']
        self.class_id_to_class_name = metadata['class_id_to_class_name']
        self.class_name_to_class_id = metadata['class_name_to_class_id']

        # captions
        self.img_id_to_encoded_caps = metadata['img_id_to_encoded_caps']
        self.word_id_to_word = metadata['word_id_to_word']
        self.word_to_word_id = metadata['word_to_word_id']
        self.pad_token     = self.word_to_word_id['<PAD>']
        self.start_token   = self.word_to_word_id['<START>']
        self.end_token     = self.word_to_word_id['<END>']
        self.unknown_token = self.word_to_word_id['<UNKNOWN>']

        self.d_vocab = metadata['num_words']
        self.num_captions_per_image = metadata['num_captions_per_image']

        nlp = English()
        self.tokenizer = nlp.Defaults.create_tokenizer(nlp) # Create a Tokenizer with the default settings for English including punctuation rules and exceptions

        # images
        if split == 'all':
            self.img_ids = metadata['img_ids']
        elif split == 'train_val':
            self.img_ids = metadata['train_val_img_ids']
        else:
            self.img_ids = metadata['test_img_ids']

    def encode_caption(self, cap):
        words = [token.text for token in self.tokenizer(cap)]
        return [self.word_to_word_id.get(word, self.unknown_token) for word in words]

    def decode_caption(self, cap):
        if isinstance(cap, torch.Tensor):
            cap = cap.tolist()
        return ' '.join([self.word_id_to_word[word_id] for word_id in cap])

    def pad_caption(self, cap):
        max_len = self.pad_to_length - 2 # 2 since we need a start token and an end token.
        cap = cap[:max_len] # truncate to maximum length.
        cap = [self.start_token] + cap + [self.end_token]
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def pad_without_start_end(self, cap):
        cap = cap[:self.pad_to_length] # truncate to maximum length.
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        class_id = self.img_id_to_class_id[img_id]
        encoded_caps = self.img_id_to_encoded_caps[img_id]
        cap_idx1, cap_idx2 = np.random.choice(self.num_captions_per_image, size=2, replace=False)
        cap1, cap2 = encoded_caps[cap_idx1], encoded_caps[cap_idx2]
        if self.should_pad:
            if self.no_start_end:
                cap1, cap_len1 = self.pad_without_start_end(cap1)
                cap2, cap_len2 = self.pad_without_start_end(cap2)
            else:
                cap1, cap_len1 = self.pad_caption(cap1)
                cap2, cap_len2 = self.pad_caption(cap2)
            return cap1, cap_len1, cap2, cap_len2
        return cap1, cap2

def get_cub_200_2011_paraphrase(d_batch=4, should_pad=False, shuffle=True, num_workers=4, **kwargs):
    dataset = CUB_200_2011_Paraphrase(should_pad=should_pad, **kwargs)
    if not should_pad:
        def collate_fn(samples):
            cap1s, cap2s = zip(*samples)
            return cap1s, cap2s
        loader = DataLoader(dataset, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=True)
    return dataset, loader

def test_cub_200_2011_paraphrase():
    dataset = CUB_200_2011_Paraphrase()
    cap1, cap2 = dataset[0]
    print(cap1, cap2)

def test_get_cub_200_2011_paraphrase():
    print('variable length unpadded:')
    dataset, loader = get_cub_200_2011_paraphrase(num_workers=0)
    batch = next(iter(loader))
    print(type(batch), len(batch))
    cap1s, cap2s = batch
    print(type(cap1s), cap1s)
    print(type(cap2s), cap2s)

    print('padded to fixed length:')
    dataset, loader = get_cub_200_2011_paraphrase(should_pad=True, pad_to_length=21, num_workers=0)
    batch = next(iter(loader))
    print(type(batch), len(batch))

    cap1s, cap_len1s, cap2s, cap_len2s = batch
    print(type(cap1s), cap1s)
    print(type(cap2s), cap2s)
    print(type(cap_len1s), cap_len1s)
    print(type(cap_len2s), cap_len2s)

##### WIP ######################################################################
##### WIP ######################################################################
##### WIP ######################################################################
##### WIP ######################################################################
##### WIP ######################################################################
##### WIP ######################################################################
##### WIP ######################################################################
##### WIP ######################################################################
##### WIP ######################################################################
##### WIP ######################################################################

class Cub200Y2011ParaphraseCombinedVocab(Dataset):
    """If should_pad is True, need to also provide a pad_to_length. Padding also adds <START> and <END> tokens to captions."""
    def __init__(self, split='all', use_10k_vocab=False, should_pad=False, pad_to_length=None, no_start_end=False, **kwargs):
        super().__init__()

        assert split in ('all', 'train_val', 'test')
        if should_pad:
            assert pad_to_length >= 3 # <START> foo <END> need at least length 3.

        self.split = split
        self.should_pad = should_pad
        self.no_start_end = no_start_end
        self.use_10k_vocab = use_10k_vocab
        self.pad_to_length = pad_to_length

        metadata = torch.load(CUB_200_2011_METADATA_PATH)

        # captions
        self.img_id_to_caps = torch.load(CUB_200_2011_IMG_ID_TO_CAPS_PATH)
        if use_10k_vocab:
            self.word_id_to_word = torch.load(WORD_ID_TO_WORD_10K_PATH)
        else:
            self.word_id_to_word = torch.load(WORD_ID_TO_WORD_5K_PATH)
        self.word_to_word_id = {v: k for k, v in self.word_id_to_word.items()}
        self.pad_token     = self.word_to_word_id['<pad>']
        self.start_token   = self.word_to_word_id['<sos>']
        self.end_token     = self.word_to_word_id['<eos>']
        self.unknown_token = self.word_to_word_id['<unk>']
        self.d_vocab = len(self.word_id_to_word)
        self.tokenizer = self.get_tokenizer()
        self.num_captions_per_image = metadata['num_captions_per_image']

        # split
        if split == 'all':
            self.img_ids = metadata['img_ids']
        elif split == 'train_val':
            self.img_ids = metadata['train_val_img_ids']
        else:
            self.img_ids = metadata['test_img_ids']

    def get_tokenizer(self):
        nlp = English()
        spacy_tokenizer = nlp.Defaults.create_tokenizer(nlp)
        tokenizer = lambda line: [token.text for token in spacy_tokenizer(line.lower())]
        return tokenizer

    def encode_caption(self, cap):
        return [self.word_to_word_id.get(word, self.unknown_token) for word in self.tokenizer(cap)]

    def decode_caption(self, cap):
        if isinstance(cap, torch.Tensor):
            cap = cap.tolist()
        return ' '.join([self.word_id_to_word[word_id] for word_id in cap])

    def pad_caption(self, cap):
        max_len = self.pad_to_length - 2 # 2 since we need a start token and an end token.
        cap = cap[:max_len] # truncate to maximum length.
        cap = [self.start_token] + cap + [self.end_token]
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def pad_without_start_end(self, cap):
        cap = cap[:self.pad_to_length] # truncate to maximum length.
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        caps = self.img_id_to_caps[img_id]
        cap_idx1, cap_idx2 = np.random.choice(self.num_captions_per_image, size=2, replace=False)
        cap1, cap2 = caps[cap_idx1], caps[cap_idx2]
        cap1, cap2 = self.encode_caption(cap1), self.encode_caption(cap2)
        if self.should_pad:
            if self.no_start_end:
                cap1, cap_len1 = self.pad_without_start_end(cap1)
                cap2, cap_len2 = self.pad_without_start_end(cap2)
            else:
                cap1, cap_len1 = self.pad_caption(cap1)
                cap2, cap_len2 = self.pad_caption(cap2)
            return cap1, cap_len1, cap2, cap_len2
        return cap1, cap2

def get_cub_200_2011_paraphrase_combined_vocab(d_batch=4, should_pad=False, shuffle=True, num_workers=4, **kwargs):
    dataset = Cub200Y2011ParaphraseCombinedVocab(should_pad=should_pad, **kwargs)
    if not should_pad:
        def collate_fn(samples):
            cap1s, cap2s = zip(*samples)
            return cap1s, cap2s
        loader = DataLoader(dataset, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=True)
    return dataset, loader

def test_get_cub_200_2011_paraphrase_combined_vocab():
    print('variable length unpadded:')
    dataset, loader = get_cub_200_2011_paraphrase_combined_vocab(num_workers=0)
    batch = next(iter(loader))
    print(type(batch), len(batch))
    cap1s, cap2s = batch
    print(type(cap1s), cap1s)
    print(type(cap2s), cap2s)

    print('padded to fixed length:')
    dataset, loader = get_cub_200_2011_paraphrase_combined_vocab(should_pad=True, pad_to_length=21, num_workers=0)
    batch = next(iter(loader))
    print(type(batch), len(batch))

    cap1s, cap_len1s, cap2s, cap_len2s = batch
    print(type(cap1s), cap1s)
    print(type(cap2s), cap2s)
    print(type(cap_len1s), cap_len1s)
    print(type(cap_len2s), cap_len2s)

#### WIP AGAIN #################################################
#### WIP AGAIN #################################################
#### WIP AGAIN #################################################
#### WIP AGAIN #################################################
#### WIP AGAIN #################################################
#### WIP AGAIN #################################################
#### WIP AGAIN #################################################
#### WIP AGAIN #################################################
#### WIP AGAIN #################################################

class QuoraParaphraseDatasetCombinedVocab(Dataset):
    """If should_pad is True, need to also provide a pad_to_length. Padding also adds <START> and <END> tokens to captions."""
    def __init__(self, paraphrases_path=QUORA_PARAPHRASE_DATA_PATH, split='train', use_10k_vocab=False, should_pad=False, pad_to_length=None, no_start_end=False, **kwargs):
        super().__init__()

        assert split in ('train', 'valid', 'test')
        if should_pad:
            assert pad_to_length >= 3 # <START> foo <END> need at least length 3.

        self.split = split
        self.should_pad = should_pad
        self.no_start_end = no_start_end
        self.use_10k_vocab = use_10k_vocab
        self.pad_to_length = pad_to_length

        self.paraphrases = torch.load(paraphrases_path)

        # captions
        if use_10k_vocab:
            self.word_id_to_word = torch.load(WORD_ID_TO_WORD_10K_PATH)
        else:
            self.word_id_to_word = torch.load(WORD_ID_TO_WORD_5K_PATH)
        self.word_to_word_id = {v: k for k, v in self.word_id_to_word.items()}
        self.pad_token     = self.word_to_word_id['<pad>']
        self.start_token   = self.word_to_word_id['<sos>']
        self.end_token     = self.word_to_word_id['<eos>']
        self.unknown_token = self.word_to_word_id['<unk>']
        self.d_vocab = len(self.word_id_to_word)
        self.tokenizer = self.get_tokenizer()

        # split
        if split == 'train':
            self.paraphrases = self.paraphrases[:100000] # first 100k as train set.
        elif split == 'test':
            self.paraphrases = self.paraphrases[100000:130000] # next 30k as valid set.
        else:
            self.paraphrases = self.paraphrases[130000:] # rest as validation set.

    def get_tokenizer(self):
        nlp = English()
        spacy_tokenizer = nlp.Defaults.create_tokenizer(nlp)
        tokenizer = lambda line: [token.text for token in spacy_tokenizer(line.lower())]
        return tokenizer

    def encode_caption(self, cap):
        return [self.word_to_word_id.get(word, self.unknown_token) for word in self.tokenizer(cap)]

    def decode_caption(self, cap):
        if isinstance(cap, torch.Tensor):
            cap = cap.tolist()
        return ' '.join([self.word_id_to_word[word_id] for word_id in cap])

    def pad_caption(self, cap):
        max_len = self.pad_to_length - 2 # 2 since we need a start token and an end token.
        cap = cap[:max_len] # truncate to maximum length.
        cap = [self.start_token] + cap + [self.end_token]
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def pad_without_start_end(self, cap):
        cap = cap[:self.pad_to_length] # truncate to maximum length.
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def __len__(self):
        return len(self.paraphrases)

    def __getitem__(self, idx):
        text1, text2 = self.paraphrases[idx]
        text1, text2 = self.encode_caption(text1), self.encode_caption(text2)
        if self.should_pad:
            if self.no_start_end:
                text1, text1_len = self.pad_without_start_end(text1)
                text2, text2_len = self.pad_without_start_end(text2)
            else:
                text1, text1_len = self.pad_caption(text1)
                text2, text2_len = self.pad_caption(text2)
            return text1, text1_len, text2, text2_len
        return text1, text2

def get_quora_paraphrase_dataset_combined_vocab(d_batch=4, should_pad=False, shuffle=True, num_workers=4, **kwargs):
    dataset = QuoraParaphraseDatasetCombinedVocab(should_pad=should_pad, **kwargs)
    if not should_pad:
        def collate_fn(samples):
            text1s, text2s = zip(*samples)
            return text1s, text2s
        loader = DataLoader(dataset, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=True)
    return dataset, loader

def test_get_quora_paraphrase_dataset_combined_vocab():
    d_batch = 5
    d_max_seq_len = 26

    dataset, loader = get_quora_paraphrase_dataset_combined_vocab(d_batch=d_batch)
    batch = next(iter(loader))
    text1s, text2s = batch
    assert len(text1s) == d_batch
    assert len(text2s) == d_batch

    dataset, loader = get_quora_paraphrase_dataset_combined_vocab(d_batch=d_batch, should_pad=True, pad_to_length=d_max_seq_len)
    batch = next(iter(loader))
    text1s, text1_lens, text2s, text2_lens = batch
    assert text1s.size() == (d_batch, d_max_seq_len) and text1s.dtype == torch.long
    assert text1_lens.size() == (d_batch,) and text1_lens.dtype == torch.long
    assert text2s.size() == (d_batch, d_max_seq_len) and text2s.dtype == torch.long
    assert text2_lens.size() == (d_batch,) and text2_lens.dtype == torch.long

def run_tests():
    print('running tests:')
#     print('test cub paraphrase:')
#     test_cub_200_2011_paraphrase()
#     test_get_cub_200_2011_paraphrase()
#     print('test combined vocab cub paraphrase:')
#     test_get_cub_200_2011_paraphrase_combined_vocab()
    print('test combined vocab quora paraphrase:')
    test_get_quora_paraphrase_dataset_combined_vocab()
    print('DONE!!')

if __name__ == '__main__':
    run_tests()
