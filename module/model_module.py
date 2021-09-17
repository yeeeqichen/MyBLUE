import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoConfig, AdamW, get_linear_schedule_with_warmup
import torch
from sklearn import metrics
from module.data_module import BLUEDataModule


class BioBERT(pl.LightningModule):
    def __init__(self, model_name_or_path,
                 num_labels,
                 task_name='ChemProt',
                 learning_rate=1e-4,
                 adam_epsilon=1e-8,
                 warmup_steps=200,
                 weight_decay=0.01,
                 train_batch_size=32,
                 eval_batch_size=32
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.model_config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = None
        if task_name == 'ChemProt':
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                            config=self.model_config)
        self.total_steps = None

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch_input, batch_idx):
        outputs = self.model(**batch_input)
        loss = outputs[0]
        return loss

    def validation_step(self, batch_input, batch_idx, data_loader_idx=0):
        outputs = self(**batch_input)
        val_loss, logits = outputs[:2]

        preds = torch.argmax(logits, axis=1)

        labels = batch_input["labels"]
        return {'loss': val_loss, "preds": preds, "labels": labels}

    def test_step(self, batch_input, batch_idx, dataloader_idx=0):
        outputs = self(**batch_input)
        _, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        labels = batch_input["labels"]
        return {"preds": preds, "labels": labels}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        precision_mic, recall_mic, f1_score_mic, _ = metrics.precision_recall_fscore_support(labels, preds,
                                                                                             average='micro')
        precision_mac, recall_mac, f1_score_mac, _ = metrics.precision_recall_fscore_support(labels, preds,
                                                                                             average='macro')
        self.log('test_precision_micro', precision_mic, prog_bar=True, sync_dist=True)
        self.log('test_recall_micro', recall_mic, prog_bar=True, sync_dist=True)
        self.log('test_f1_micro', f1_score_mic, prog_bar=True, sync_dist=True)
        self.log('test_precision_macro', precision_mac, prog_bar=True, sync_dist=True)
        self.log('test_recall_macro', recall_mac, prog_bar=True, sync_dist=True)
        self.log('test_f1_macro', f1_score_mac, prog_bar=True, sync_dist=True)
        return None

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        # precision_mic, recall_mic, f1_score_mic, _ = metrics.precision_recall_fscore_support(labels, preds,
        #                                                                                      average='micro')
        precision_mac, recall_mac, f1_score_mac, _ = metrics.precision_recall_fscore_support(labels, preds,
                                                                                             average='macro')
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        # self.log('val_precision_micro', precision_mic, prog_bar=True, sync_dist=True)
        # self.log('val_recall_micro', recall_mic, prog_bar=True, sync_dist=True)
        # self.log('val_f1_micro', f1_score_mic, prog_bar=True, sync_dist=True)
        self.log('val_precision_macro', precision_mac, prog_bar=True, sync_dist=True)
        self.log('val_recall_macro', recall_mac, prog_bar=True, sync_dist=True)
        self.log('val_f1_macro', f1_score_mac, prog_bar=True, sync_dist=True)
        return loss

    def setup(self, stage=None):
        if stage != 'fit':
            return
        train_dataloader = self.train_dataloader()
        tb_size = self.hparams.train_batch_size * max(1, len(self.trainer.gpus.split(',')))
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_dataloader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]


# datamodule = BLUEDataModule(train_file='../data/ChemProt/train.tsv', valid_file='../data/ChemProt/dev.tsv')
# datamodule.setup()
# model = BioBERT(
#     model_name_or_path='dmis-lab/biobert-base-cased-v1.2',
#     num_labels=6,
# )
# trainer = pl.Trainer(max_epochs=1, gpus=-1)
# trainer.fit(model, datamodule)
from transformers import BertForTokenClassification
model = BertForTokenClassification.from_pretrained('')
model.state_dict()