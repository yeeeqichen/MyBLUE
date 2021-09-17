import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch

label_map = {
    'ChemProt': {
        'false': 0,
        'CPR:3': 1,
        'CPR:4': 2,
        'CPR:5': 3,
        'CPR:6': 4,
        'CPR:9': 5
    }
}


class MyDataset(Dataset):
    def __init__(self, file_path, task='ChemProt'):
        print('reading data from {}'.format(file_path))
        self.dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                if index == 0:
                    continue
                _, sentence, label = line.strip('\n').split('\t')
                label_id = label_map[task][label]
                self.dataset.append([sentence, label_id])
        print('total instance:', len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


class BLUEDataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path='dmis-lab/biobert-base-cased-v1.2',
                 train_batch_size=32,
                 valid_batch_size=32,
                 test_batch_size=32,
                 max_seq_length=128,
                 train_file='',
                 valid_file='',
                 test_file='',
                 num_workers=4):
        super(BLUEDataModule, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.max_seq_length = max_seq_length
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.train_dataset = None
        self.valid_dataset = None
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.train_batch_size,
                          collate_fn=self.convert_to_features,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.valid_batch_size,
                          collate_fn=self.convert_to_features,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.test_batch_size,
                          collate_fn=self.convert_to_features,
                          num_workers=self.num_workers)

    def convert_to_features(self, batch_input):
        sentences = [_[0] for _ in batch_input]
        labels = [_[1] for _ in batch_input]
        encode_results = self.tokenizer.batch_encode_plus(
            sentences,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
        )
        features = dict()
        for key in encode_results.keys():
            features[key] = torch.LongTensor(encode_results[key])
        features['labels'] = torch.LongTensor(labels)

        return features

    def setup(self, stage=None):
        self.train_dataset = MyDataset(file_path=self.train_file)
        self.valid_dataset = MyDataset(file_path=self.valid_file)
        self.test_dataset = MyDataset(file_path=self.test_file)


# datamodule = BLUEDataModule(train_file='../data/ChemProt/train.tsv', valid_file='../data/ChemProt/dev.tsv')
# datamodule.setup()
# train_dataloader = datamodule.train_dataloader()
# for _ in train_dataloader:
#     print(_)
#     break
