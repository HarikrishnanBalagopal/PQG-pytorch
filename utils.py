import os
import math
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from spacy.lang.en import English
from torchvision import transforms
from torchvision.models import inception_v3
from torch.utils.data import Dataset, DataLoader

DATASETS_PATH = '/gdrive/My Drive/datasets/'

# CUB_200_2011
CUB_200_2011_PATH = os.path.join(DATASETS_PATH, 'cub_200_2011')
# Captions and metadata
CUB_200_2011_METADATA_PATH           = os.path.join(CUB_200_2011_PATH, 'metadata.pth')
CUB_200_2011_IMG_ID_TO_CAPS_PATH     = os.path.join(CUB_200_2011_PATH, 'cub_200_2011_img_id_to_caps.pth')
# All images
CUB_200_2011_IMGS_64_PATH            = os.path.join(CUB_200_2011_PATH, 'imgs_64x64.pth')
CUB_200_2011_IMGS_128_PATH           = os.path.join(CUB_200_2011_PATH, 'imgs_128x128.pth')
CUB_200_2011_IMGS_256_PATH           = os.path.join(CUB_200_2011_PATH, 'imgs_256x256.pth')
# Training/Validation split images
CUB_200_2011_TRAIN_VAL_IMGS_64_PATH  = os.path.join(CUB_200_2011_PATH, 'imgs_train_val_64x64.pth')
CUB_200_2011_TRAIN_VAL_IMGS_128_PATH = os.path.join(CUB_200_2011_PATH, 'imgs_train_val_128x128.pth')
CUB_200_2011_TRAIN_VAL_IMGS_256_PATH = os.path.join(CUB_200_2011_PATH, 'imgs_train_val_256x256.pth')
# Testing split images
CUB_200_2011_TEST_IMGS_64_PATH       = os.path.join(CUB_200_2011_PATH, 'imgs_test_64x64.pth')
CUB_200_2011_TEST_IMGS_128_PATH      = os.path.join(CUB_200_2011_PATH, 'imgs_test_128x128.pth')
CUB_200_2011_TEST_IMGS_256_PATH      = os.path.join(CUB_200_2011_PATH, 'imgs_test_256x256.pth')
# GLOVE word embeddings for CUB 200 2011
CUB_200_2011_GLOVE_PATH = os.path.join(CUB_200_2011_PATH, 'glove_relevant_embeddings.pth')
CUB_200_2011_D_VOCAB = 1750

# OXFORD_FLOWERS_102
OXFORD_FLOWERS_102_PATH = os.path.join(DATASETS_PATH, 'oxford_flowers_102')
# Captions and metadata
OXFORD_FLOWERS_102_METADATA_PATH           = os.path.join(OXFORD_FLOWERS_102_PATH, 'metadata.pth')
# All images
OXFORD_FLOWERS_102_IMGS_64_PATH            = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_64x64.pth')
OXFORD_FLOWERS_102_IMGS_128_PATH           = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_128x128.pth')
OXFORD_FLOWERS_102_IMGS_256_PATH           = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_256x256.pth')
# Training/Validation split images
OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_64_PATH  = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_train_val_64x64.pth')
OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_128_PATH = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_train_val_128x128.pth')
OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_256_PATH = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_train_val_256x256.pth')
# Testing split images
OXFORD_FLOWERS_102_TEST_IMGS_64_PATH       = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_test_64x64.pth')
OXFORD_FLOWERS_102_TEST_IMGS_128_PATH      = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_test_128x128.pth')
OXFORD_FLOWERS_102_TEST_IMGS_256_PATH      = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_test_256x256.pth')

# EMNLP_2017_NEWS
EMNLP_2017_NEWS_PATH = os.path.join(DATASETS_PATH, 'emnlp_2017_news')
EMNLP_2017_NEWS_DATA_PATH = os.path.join(EMNLP_2017_NEWS_PATH, 'data.pth')

# GLOVE 300 DIM EMBEDDINGS OF 5102 WORDS (from scratchgan paper)
GLOVE_5102_300_PATH = os.path.join(DATASETS_PATH, 'glove_5102_300.pth')

# CUB 200 2011, OXFORD FLOWERS 102, EMNLP 2017 AND WIKITEXT 103 COMBINED VOCAB
COMBINED_WORD_FREQ_PATH = os.path.join(DATASETS_PATH, 'cub_oxford_emnlp_wikitext_word_freqs.pth')
WORD_ID_TO_WORD_5K_PATH = os.path.join(DATASETS_PATH, 'word_id_to_word_5k.pth')
WORD_ID_TO_WORD_10K_PATH = os.path.join(DATASETS_PATH, 'word_id_to_word_10k.pth')

class CUB_200_2011(Dataset):
    """If should_pad is True, need to also provide a pad_to_length. Padding also adds <START> and <END> tokens to captions."""
    def __init__(self, split='all', d_image_size=64, transform=None, should_pad=False, pad_to_length=None, no_start_end=False, **kwargs):
        super().__init__()

        assert split in ('all', 'train_val', 'test')
        assert d_image_size in (64, 128, 256)
        if should_pad:
            assert pad_to_length >= 3 # <START> foo <END> need at least length 3.

        self.split = split
        self.d_image_size = d_image_size
        self.transform = transform
        self.should_pad = should_pad
        self.pad_to_length = pad_to_length
        self.no_start_end = no_start_end

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
            if d_image_size == 64:
                imgs_path = CUB_200_2011_IMGS_64_PATH
            elif d_image_size == 128:
                imgs_path = CUB_200_2011_IMGS_128_PATH
            else:
                imgs_path = CUB_200_2011_IMGS_256_PATH
        elif split == 'train_val':
            self.img_ids = metadata['train_val_img_ids']
            if d_image_size == 64:
                imgs_path = CUB_200_2011_TRAIN_VAL_IMGS_64_PATH
            elif d_image_size == 128:
                imgs_path = CUB_200_2011_TRAIN_VAL_IMGS_128_PATH
            else:
                imgs_path = CUB_200_2011_TRAIN_VAL_IMGS_256_PATH
        else:
            self.img_ids = metadata['test_img_ids']
            if d_image_size == 64:
                imgs_path = CUB_200_2011_TEST_IMGS_64_PATH
            elif d_image_size == 128:
                imgs_path = CUB_200_2011_TEST_IMGS_128_PATH
            else:
                imgs_path = CUB_200_2011_TEST_IMGS_256_PATH

        self.imgs = torch.load(imgs_path)
        assert self.imgs.size()[1:] == (3, d_image_size, d_image_size) and self.imgs.dtype == torch.uint8

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
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            img = self.transform(img)
        img_id = self.img_ids[idx]
        class_id = self.img_id_to_class_id[img_id]
        encoded_caps = self.img_id_to_encoded_caps[img_id]
        cap_idx = torch.randint(low=0, high=self.num_captions_per_image, size=(1,)).item()
        encoded_cap = encoded_caps[cap_idx]
        if self.should_pad:
            if self.no_start_end:
                encoded_cap, cap_len = self.pad_without_start_end(encoded_cap)
            else:
                encoded_cap, cap_len = self.pad_caption(encoded_cap)
            return img, class_id, encoded_cap, cap_len
        return img, class_id, encoded_cap

class OxfordFlowers102(Dataset):
    """If should_pad is True, need to also provide a pad_to_length. Padding also adds <START> and <END> tokens to captions."""
    def __init__(self, split='all', d_image_size=64, transform=None, should_pad=False, pad_to_length=None, no_start_end=False, **kwargs):
        super().__init__()

        assert split in ('all', 'train_val', 'test')
        assert d_image_size in (64, 128, 256)
        if should_pad:
            assert pad_to_length >= 3 # <START> foo <END> need at least length 3.

        self.split = split
        self.d_image_size = d_image_size
        self.transform = transform
        self.should_pad = should_pad
        self.pad_to_length = pad_to_length
        self.no_start_end = no_start_end

        metadata = torch.load(OXFORD_FLOWERS_102_METADATA_PATH)

        # labels
        self.img_id_to_class_id = metadata['img_id_to_class_id']
        # credit https://github.com/jimgoo/caffe-oxford102/blob/master/class_labels.py
        self.class_id_to_class_name = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']
        self.class_name_to_class_id = {c: i for i, c in enumerate(self.class_id_to_class_name)}

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
            if d_image_size == 64:
                imgs_path = OXFORD_FLOWERS_102_IMGS_64_PATH
            elif d_image_size == 128:
                imgs_path = OXFORD_FLOWERS_102_IMGS_128_PATH
            else:
                imgs_path = OXFORD_FLOWERS_102_IMGS_256_PATH
        elif split == 'train_val':
            self.img_ids = metadata['train_val_img_ids']
            if d_image_size == 64:
                imgs_path = OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_64_PATH
            elif d_image_size == 128:
                imgs_path = OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_128_PATH
            else:
                imgs_path = OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_256_PATH
        else:
            self.img_ids = metadata['test_img_ids']
            if d_image_size == 64:
                imgs_path = OXFORD_FLOWERS_102_TEST_IMGS_64_PATH
            elif d_image_size == 128:
                imgs_path = OXFORD_FLOWERS_102_TEST_IMGS_128_PATH
            else:
                imgs_path = OXFORD_FLOWERS_102_TEST_IMGS_256_PATH

        self.imgs = torch.load(imgs_path)
        assert self.imgs.size()[1:] == (3, d_image_size, d_image_size) and self.imgs.dtype == torch.uint8

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
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            img = self.transform(img)
        img_id = self.img_ids[idx]
        class_id = self.img_id_to_class_id[img_id]
        encoded_caps = self.img_id_to_encoded_caps[img_id]
        cap_idx = torch.randint(low=0, high=self.num_captions_per_image, size=(1,)).item()
        encoded_cap = encoded_caps[cap_idx]
        if self.should_pad:
            if self.no_start_end:
                encoded_cap, cap_len = self.pad_without_start_end(encoded_cap)
            else:
                encoded_cap, cap_len = self.pad_caption(encoded_cap)
            return img, class_id, encoded_cap, cap_len
        return img, class_id, encoded_cap

class EMNLP2017(Dataset):
    def __init__(self, dataset_path=EMNLP_2017_NEWS_DATA_PATH, split='train', **kwargs):
        assert split in ('train', 'valid', 'test')
        assert os.path.isfile(dataset_path)

        data = torch.load(dataset_path)

        self.split = split
        self.word_id_to_word = data['word_id_to_word']
        self.word_to_word_id = data['word_to_word_id']
        if split == 'train':
            self.seqs = data['train_data_sequences'].long()
            self.seq_lens = data['train_data_sequence_lengths'].long()
        elif split == 'valid':
            self.seqs = data['valid_data_sequences'].long()
            self.seq_lens = data['valid_data_sequence_lengths'].long()
        else:
            self.seqs = data['test_data_sequences'].long()
            self.seq_lens = data['test_data_sequence_lengths'].long()

        self.d_max_seq_len = self.seqs.size(1)
        self.d_vocab = len(self.word_id_to_word)
        self.pad_token = self.word_to_word_id[' ']
        self.unknown_token = self.word_to_word_id['<unk>']

    def encode(self, text):
        return [self.word_to_word_id.get(word, self.unknown_token) for word in text.split()]

    def decode(self, text):
        if isinstance(text, torch.Tensor):
            text = text.tolist()
        return ' '.join([self.word_id_to_word[word_id] for word_id in text])

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_lens[idx]

    def __len__(self):
        return len(self.seqs)

def to_one_hot(labels, d_classes):
    """
    Args:
        labels (Tensor): integer tensor of shape (d_batch, *)
        d_classes (int): number of classes
    Returns:
        (Tensor): float tensor of shape (d_batch, *, d_classes), one hot representation of the labels
    """
    return torch.zeros(*labels.size(), d_classes, device=labels.device).scatter_(-1, labels.unsqueeze(-1), 1)

def get_cub_200_2011(d_batch=4, should_pad=False, shuffle=True, **kwargs):
    transform = transforms.Lambda(lambda x: (x.float() / 255.) * 2. - 1.)
    train_set = CUB_200_2011(transform=transform, should_pad=should_pad, **kwargs)
    if not should_pad:
        def collate_fn(samples):
            imgs, class_ids, caps = zip(*samples)
            imgs = torch.stack(imgs)
            class_ids = torch.tensor(class_ids, dtype=torch.long)
            return imgs, class_ids, caps
        train_loader = DataLoader(train_set, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=4, pin_memory=True)
    return train_set, train_loader

def get_oxford_flowers_102(d_batch=4, should_pad=False, shuffle=True, **kwargs):
    transform = transforms.Lambda(lambda x: (x.float() / 255.) * 2. - 1.)
    train_set = OxfordFlowers102(transform=transform, should_pad=should_pad, **kwargs)
    if not should_pad:
        def collate_fn(samples):
            imgs, class_ids, caps = zip(*samples)
            imgs = torch.stack(imgs)
            class_ids = torch.tensor(class_ids, dtype=torch.long)
            return imgs, class_ids, caps
        train_loader = DataLoader(train_set, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=4, pin_memory=True)
    return train_set, train_loader

def get_emnlp_2017(d_batch=4, shuffle=True, **kwargs):
    dataset = EMNLP2017(**kwargs)
    loader = DataLoader(dataset, batch_size=d_batch, shuffle=shuffle, drop_last=True, num_workers=4, pin_memory=True)
    return dataset, loader

###################################### TESTS ######################################

def test_get_cub_200_2011():
    train_set, train_loader = get_cub_200_2011(should_pad=True, pad_to_length=25)
    print('Size of vocabulary:', train_set.d_vocab)
    print('Number of images:', len(train_set))
    print('Number of batches:', len(train_loader))
    batch = next(iter(train_loader))
#     imgs, caps = batch
#     print(imgs.size(), imgs.dtype, imgs.max(), imgs.min())
#     print(caps)
    imgs, class_ids, caps, cap_lens = batch
    print(imgs.size(), imgs.dtype, imgs.min(), imgs.max())
    print(class_ids.size(), class_ids.dtype, class_ids)
    print(caps.size(), caps.dtype, caps)
    print(cap_lens.size(), cap_lens.dtype, cap_lens)
    for cap in caps:
        print(train_set.decode_caption(cap))

    # visualize
    denorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    imgs = torch.stack([denorm(img) for img in imgs]).permute(0, 2, 3, 1).cpu()
    plt.figure(figsize=(1*8,4*8))
    for i, img in enumerate(imgs):
        plt.subplot(4, 1, i+1)
        plt.imshow(img)
        plt.axis('off')
        cap = train_set.decode_caption(caps[i])
        plt.title(cap)
    plt.suptitle('some images from the dataset')

def test_oxford_flowers_102():
    train_set, train_loader = get_oxford_flowers_102(should_pad=True, pad_to_length=25)
    print('Size of vocabulary:', train_set.d_vocab)
    print('Number of images:', len(train_set))
    print('Number of batches:', len(train_loader))
    batch = next(iter(train_loader))
#     imgs, caps = batch
#     print(imgs.size(), imgs.dtype, imgs.max(), imgs.min())
#     print(caps)
    imgs, class_ids, caps, cap_lens = batch
    print(imgs.size(), imgs.dtype, imgs.min(), imgs.max())
    print(class_ids.size(), class_ids.dtype, class_ids)
    print(caps.size(), caps.dtype, caps)
    print(cap_lens.size(), cap_lens.dtype, cap_lens)
    for cap in caps:
        print(train_set.decode_caption(cap))

    # visualize
    denorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    imgs = torch.stack([denorm(img) for img in imgs]).permute(0, 2, 3, 1).cpu()
    plt.figure(figsize=(1*8,4*8))
    for i, img in enumerate(imgs):
        plt.subplot(4, 1, i+1)
        plt.imshow(img)
        plt.axis('off')
        cap = train_set.decode_caption(caps[i])
        plt.title(cap)
    plt.suptitle('some images from the dataset')

def test_get_emnlp_2017():
    d_batch = 16
    dataset, loader = get_emnlp_2017(d_batch=d_batch)
    batch = next(iter(loader))
    texts, text_lens = batch
    assert texts.size() == (d_batch, dataset.d_max_seq_len) and texts.dtype == torch.long
    assert text_lens.size() == (d_batch,) and text_lens.dtype == torch.long

def mkdir_p(path):
    """Make directory and all necessary parent directories given by the path."""
    os.makedirs(path, exist_ok=True)

def get_timestamp():
    """Return the current date and time in year, month, day, hour, minute and second format."""
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_%H_%M_%S')

def uniform_in_range(low=0., high=1., size=(1,), **kwargs):
    return (high - low) * torch.rand(*size, **kwargs) + low

def get_biased_coin(probability_of_heads=0.5):
    """Get a biased coin that returns True on heads."""
    return lambda: torch.rand(1).item() < probability_of_heads

def get_cub_200_2011_glove_embeddings(fine_tune=False):
    GLOVE_D_EMBED = 50
    glove_embeddings = torch.load(CUB_200_2011_GLOVE_PATH)
    assert glove_embeddings.size() == (CUB_200_2011_D_VOCAB, GLOVE_D_EMBED)
    assert glove_embeddings.dtype == torch.float
    embed = nn.Embedding(CUB_200_2011_D_VOCAB, GLOVE_D_EMBED)
    embed.weight == nn.Parameter(glove_embeddings)
    embed.weight.requires_grad = fine_tune

    return embed

def text_embeds_to_encoded_texts(text_embeds, ref_embeds):
    """
    Find closest words using euclidean distance.
    Args:
        text_embeds (Tensor): float tensor of shape (d_batch, d_max_seq_len, d_text_embed)
        ref_embeds (Tensor): float tensor of shape (d_vocab, d_text_embed)
    Returns:
        encoded_texts list(list(int)): list of list of word indices.
    """
    encoded_texts = []
    for text_embed in text_embeds:
        encoded_text = []
        for word_embed in text_embed:
            errs = (ref_embeds - word_embed.unsqueeze(0))
            dists = (errs**2).sum(dim=1)
            word_id = dists.argmin().item()
            encoded_text.append(word_id)
        encoded_texts.append(encoded_text)
    return encoded_texts

def damsm_loss(img_local_features, img_global_features, text_local_features, text_global_features, visualize=False, gamma_1=5.0, gamma_2=5.0, gamma_3=10.0):
    """
    Deep Attentional Multimodal Similarity Model (DAMSM) as given in AttnGAN paper (https://arxiv.org/pdf/1711.10485.pdf)
    Calculate the loss. (Based on https://github.com/taoxugit/AttnGAN/blob/master/code/miscc/losses.py)
    NOTE: d_common = d_text_feature
    Args:
        img_local_features (Tensor): float tensor of shape (d_batch, d_img_regions, d_common)
        img_global_features (Tensor): float tensor of shape (d_batch, d_common)
        text_local_features (Tensor): float tensor of shape (d_batch, d_max_seq_len, d_common)
        text_global_features (Tensor): float tensor of shape (d_batch, d_common)
    Returns:
        loss (Tensor): float tensor scalar.
    """
    device = img_local_features.device
    d_batch = img_local_features.size(0)
    R_Q_Ds = torch.empty(d_batch, d_batch, device=device)
    R_Q_D_globals = torch.empty(d_batch, d_batch, device=device)

    for i in range(d_batch):
        R_Q_Ds[i] = loss_helper(
            img_local_features,
            text_local_features[i].unsqueeze(0).repeat(d_batch, 1, 1),
            gamma_1=gamma_1, gamma_2=gamma_2
        )
        R_Q_D_globals[i] = F.cosine_similarity(img_global_features, text_global_features[i].unsqueeze(0).repeat(d_batch, 1), dim=1) # (d_batch, d_common), (d_batch, d_common) -> (d_batch,)

    P_D_Qs = F.softmax(R_Q_Ds * gamma_3, dim=0) # (d_batch, d_batch)
    P_Q_Ds = F.softmax(R_Q_Ds * gamma_3, dim=1) # (d_batch, d_batch)
    L_w_1 = -P_D_Qs.diag().log().sum()
    L_w_2 = -P_Q_Ds.diag().log().sum()

    P_D_Q_globals = F.softmax(R_Q_D_globals * gamma_3, dim=0) # (d_batch, d_batch)
    P_Q_D_globals = F.softmax(R_Q_D_globals * gamma_3, dim=1) # (d_batch, d_batch)
    L_s_1 = -P_D_Q_globals.diag().log().sum()
    L_s_2 = -P_Q_D_globals.diag().log().sum()

    loss = L_w_1 + L_w_2 + L_s_1 + L_s_2
    if visualize:
        return loss, R_Q_Ds, R_Q_D_globals, P_D_Qs, P_Q_Ds, P_D_Q_globals, P_Q_D_globals
    return loss

def loss_helper(img_local_features, text_local_features, gamma_1, gamma_2):
    ss = torch.bmm(text_local_features, img_local_features.transpose(1, 2)) # (d_batch, d_max_seq_len, d_common), (d_batch, d_img_regions, d_common) -> (d_batch, d_max_seq_len, d_img_regions)
    s_bars = F.softmax(ss, dim=1) # (d_batch, d_max_seq_len, d_img_regions) -> (d_batch, d_max_seq_len, d_img_regions)
    alphas = F.softmax(s_bars * gamma_1, dim=2) # (d_batch, d_max_seq_len, d_img_regions) -> (d_batch, d_max_seq_len, d_img_regions)
    cs = torch.bmm(alphas, img_local_features) # (d_batch, d_max_seq_len, d_img_regions), (d_batch, d_img_regions, d_common) -> (d_batch, d_max_seq_len, d_common)
    r_c_es = F.cosine_similarity(cs, text_local_features, dim=2) # (d_batch, d_max_seq_len, d_common), (d_batch, d_max_seq_len, d_common) -> (d_batch, d_max_seq_len)
    r_q_ds = (r_c_es * gamma_2).exp().sum(dim=1).log() # (d_batch, d_max_seq_len) -> (d_batch,) # TODO: raise to power 1/gamma_2 ?
    return r_q_ds

def test_damsm_loss():
    d_batch = 16
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    d_img_regions = 17*17
    d_img_local_feature = 768
    d_img_global_feature = 2048
    d_max_seq_len = 23
    d_text_local_feature = 128
    d_text_global_feature = 128
    d_common = d_text_local_feature

    img_local_features = torch.randn(d_batch, d_img_regions, d_img_local_feature).to(device)
    img_global_features = torch.randn(d_batch, d_img_global_feature).to(device)
    text_local_features = torch.randn(d_batch, d_max_seq_len, d_text_local_feature).to(device)
    text_global_features = torch.randn(d_batch, d_text_global_feature).to(device)

    m1 = nn.Linear(d_img_local_feature, d_common).to(device).train()
    m2 = nn.Linear(d_img_global_feature, d_common).to(device).train()

    img_local_features = m1(img_local_features)
    img_global_features = m2(img_global_features)

    loss = damsm_loss(img_local_features, img_global_features, text_local_features, text_global_features)
    assert math.isfinite(loss)
    loss.backward()
    print('loss:', loss.item())

class DAMSMImageEncoder(nn.Module):
    def __init__(self, d_common=256, **kwargs):
        super().__init__()
        self.d_img_regions = 17*17
        self.d_common = d_common

        model = inception_v3(pretrained=True)
        # url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        # print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.conv_local_features = nn.Conv2d(768, self.d_common, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc_global_features = nn.Linear(2048, self.d_common)

    def init_trainable_weights(self):
        initrange = 0.1
        self.conv_local_features.weight.data.uniform_(-initrange, initrange)
        self.fc_global_features.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        img_local_features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        img_global_features = self.fc_global_features(x)
        # 512
        img_local_features = self.conv_local_features(img_local_features)
        img_local_features = img_local_features.view(-1, self.d_common, self.d_img_regions).transpose(1, 2).contiguous()
        return img_local_features, img_global_features

def test_damsm_image_encoder():
    d_batch = 4
    d_image_size = 64
    img_enc = DAMSMImageEncoder().cuda().train()
    imgs = torch.randn(d_batch, 3, d_image_size, d_image_size).cuda()

    img_local_features, img_global_features = img_enc(imgs)

    assert img_local_features.size() == (d_batch, img_enc.d_img_regions, img_enc.d_common)
    assert img_global_features.size() == (d_batch, img_enc.d_common)

"""# DAMSM text encoder"""

class DAMSMTextEncoder(nn.Module):
    def __init__(self, d_vocab=1750, d_max_seq_len=23, d_text_embed=300, d_text_local_feature=256, drop_prob=0.5, **kwargs):
        super().__init__()

        assert d_text_local_feature % 2 == 0

        self.d_vocab = d_vocab
        self.drop_prob = drop_prob
        self.d_text_embed = d_text_embed
        self.d_max_seq_len = d_max_seq_len
        self.d_text_local_feature = d_text_local_feature
        self.d_text_global_feature = d_text_local_feature
        self.d_common = d_text_local_feature

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.embed = nn.Embedding(self.d_vocab, self.d_text_embed)
        self.drop = nn.Dropout(self.drop_prob)
        self.rnn = nn.LSTM(self.d_text_embed, self.d_text_local_feature//2, batch_first=True, bidirectional=True)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, texts):
        text_embeds = self.drop(self.embed(texts))
        text_local_features, (text_global_features, _) = self.rnn(text_embeds)
        text_global_features = text_global_features.transpose(0, 1).reshape(-1, self.d_text_local_feature)
        return text_local_features, text_global_features

def test_damsm_text_encoder():
    d_batch = 4
    text_enc = DAMSMTextEncoder().cuda().train()
    texts = torch.randint(low=0, high=text_enc.d_vocab, size=(d_batch, text_enc.d_max_seq_len)).cuda()

    text_local_features, text_global_features = text_enc(texts)

    assert text_local_features.size() == (d_batch, text_enc.d_max_seq_len, text_enc.d_common)
    assert text_global_features.size() == (d_batch, text_enc.d_common)

def test_encoders_and_loss():
    d_batch = 4
    d_image_size = 64

    img_enc = DAMSMImageEncoder().cuda().train()
    text_enc = DAMSMTextEncoder().cuda().train()

    assert img_enc.d_common == text_enc.d_common

    imgs = torch.randn(d_batch, 3, d_image_size, d_image_size).cuda()
    texts = torch.randint(low=0, high=text_enc.d_vocab, size=(d_batch, text_enc.d_max_seq_len)).cuda()

    img_local_features, img_global_features = img_enc(imgs)
    text_local_features, text_global_features = text_enc(texts)

    assert img_local_features.size() == (d_batch, img_enc.d_img_regions, img_enc.d_common)
    assert img_global_features.size() == (d_batch, img_enc.d_common)
    assert text_local_features.size() == (d_batch, text_enc.d_max_seq_len, text_enc.d_common)
    assert text_global_features.size() == (d_batch, text_enc.d_common)

    loss = damsm_loss(img_local_features, img_global_features, text_local_features, text_global_features)
    assert math.isfinite(loss)
    loss.backward()
    print('loss:', loss.item())

    with torch.no_grad():
        loss, R_Q_Ds, R_Q_D_globals, P_D_Qs, P_Q_Ds, P_D_Q_globals, P_Q_D_globals = damsm_loss(img_local_features, img_global_features, text_local_features, text_global_features, visualize=True)
        print('loss:', loss.item())
        for xs in [R_Q_Ds, R_Q_D_globals, P_D_Qs, P_Q_Ds, P_D_Q_globals, P_Q_D_globals]:
            assert xs.size() == (d_batch, d_batch)

if __name__ == '__main__':
    print('Running tests for CUB 200 2011 dataset:')
    test_get_cub_200_2011()
    print('Running tests for OxfordFlowers102 dataset:')
    test_oxford_flowers_102()
    print('Running tests for EMNLP2017 dataset:')
    test_get_emnlp_2017()
    print('Running tests for DAMSM loss and encoders:')
    test_damsm_loss()
    test_damsm_image_encoder()
    test_damsm_text_encoder()
    test_encoders_and_loss()
    print('Done.')
