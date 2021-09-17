import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import re

label_map = {
    'ChemProt': {
        'false': 0,
        'CPR:3': 1,
        'CPR:4': 2,
        'CPR:5': 3,
        'CPR:6': 4,
        'CPR:9': 5
    },
    'DDI': {
        'DDI-false': 0,
        'DDI-mechanism': 1,
        'DDI-effect': 2,
        'DDI-advise': 3,
        'DDI-int': 4,
    },
    'BC5CDR': {
        'O': 0,
        'I': 2,
        'B': 1
    }
}


class MyDataset(Dataset):
    def __init__(self, file_path, task='ChemProt'):
        print('reading data from {}'.format(file_path))
        self.dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            if task in ['ChemProt', 'DDI']:
                for index, line in enumerate(f):
                    if index == 0:
                        continue
                    _, sentence, label = line.strip('\n').split('\t')
                    label_id = label_map[task][label]
                    self.dataset.append([sentence, label_id])
            elif task == 'BC5CDR':
                flag = True
                while flag:
                    flag, words, labels = self._parse_ner_data(f)
                    self.dataset.append([words, labels])
        print('total instance:', len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    @staticmethod
    def _parse_ner_data(fd):
        sent1 = fd.readline().strip('\n')
        if sent1 == '':
            return False, None, None
        sent2 = fd.readline().strip('\n')
        sent1 = re.match('([0-9]*)(\|t\|)(.*)', sent1).groups()[2]
        sent2 = re.match('([0-9]*)(\|a\|)(.*)', sent2).groups()[2]
        sentence = sent1 + ' ' + sent2
        visit_map = [0] * len(sentence)
        for line in fd:
            if line == '\n':
                break
            if len(line.split('\t')) == 4:
                continue
            start, end, entity = line.split('\t')[1:4]
            for i in range(int(start), int(end)):
                visit_map[i] = 1
        pre_pos = 0
        next_label = 'O'
        words = []
        labels = []
        for i in range(len(sentence)):
            char = sentence[i]
            flag = visit_map[i]
            if char == ' ':
                words.append(sentence[pre_pos: i])
                labels.append(next_label)
                pre_pos = i + 1
                if flag == 1:
                    next_label = 'I'
                else:
                    next_label = 'O'
            else:
                if flag == 1:
                    if next_label != 'I':
                        next_label = 'B'
        return True, words, labels


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
                 task='ChemProt',
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
        self.test_dataset = None
        self.num_workers = num_workers
        self.task = task

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
        features = None
        if self.task in ['ChemProt', 'DDI']:
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
        else:
            batch_words = [_[0] for _ in batch_input]
            batch_labels = [_[1] for _ in batch_input]
            batch_tokens = []
            batch_label_ids = []
            for words, labels in zip(batch_words, batch_labels):
                tokens = []
                label_ids = []
                for word, label in zip(words, labels):
                    word_tokens = self.tokenizer.tokenize(word)
                    if len(word_tokens) > 0:
                        tokens.extend(word_tokens)
                        label_ids.extend([label_map[self.task][label]] + [3] * (len(word_tokens) - 1))
                batch_tokens.append(tokens)
                batch_label_ids.append(label_ids)
            # todo: padding to max_seq_length and add special token like '[CLS]'
            pass

        return features

    def setup(self, stage=None):
        self.train_dataset = MyDataset(file_path=self.train_file, task=self.task)
        self.valid_dataset = MyDataset(file_path=self.valid_file, task=self.task)
        self.test_dataset = MyDataset(file_path=self.test_file, task=self.task)


# datamodule = BLUEDataModule(train_file='../data/ChemProt/train.tsv', valid_file='../data/ChemProt/dev.tsv')
# datamodule.setup()
# train_dataloader = datamodule.train_dataloader()
# for _ in train_dataloader:
#     print(_)
#     break
