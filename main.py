from module.data_module import BLUEDataModule
from module.model_module import BioBERT
from pytorch_lightning import Trainer
import argparse

num_labels = {
    'ChemProt': 6,
    'DDI': 5,
}

task_list = ['ChemProt', 'DDI']

parser = argparse.ArgumentParser()

# path parameter
parser.add_argument('--model_name_or_path', type=str, default='',
                    help='')
parser.add_argument('--train_file', type=str, default='',
                    help='')
parser.add_argument('--valid_file', type=str, default='',
                    help='')
parser.add_argument('--test_file', type=str, default='',
                    help='')
parser.add_argument('--output_dir', type=str, default='',
                    help='')
# training configuration
parser.add_argument('--batch_size', type=int, default=52)
parser.add_argument('--train_batch_size', type=int, default=32,
                    help='')
parser.add_argument('--valid_batch_size', type=int, default=32,
                    help='')
parser.add_argument('--test_batch_size', type=int, default=32,
                    help='')
parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='')
parser.add_argument('--warm_steps', type=float, default=500,
                    help='')
parser.add_argument('--weight_decay', type=float, default=0.01,
                    help='')
parser.add_argument('--log_steps', type=int, default=40,
                    help='')
parser.add_argument('--epochs', type=int, default=10,
                    help='')
parser.add_argument('--max_seq_length', type=int, default=128,
                    help='')
parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                    help='')

# other settings
parser.add_argument('--task', type=str, required=True,
                    help='')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--dataloader_workers', type=int, default=4,
                    help='')

args = parser.parse_args()

if args.task not in task_list:
    print('task name: ', args.task)
    print('please specify a valid task name!, only support: ', task_list)
    exit('invalid task name')


def get_batch_size(purpose='train'):
    if args.batch_size != 0:
        return args.batch_size
    if purpose == 'train':
        return args.train_batch_size
    if purpose == 'valid':
        return args.valid_batch_size
    if purpose == 'test':
        return args.test_batch_size


data_module = BLUEDataModule(
    model_name_or_path=args.model_name_or_path,
    train_file=args.train_file,
    valid_file=args.valid_file,
    test_file=args.test_file,
    train_batch_size=get_batch_size('train'),
    valid_batch_size=get_batch_size('valid'),
    test_batch_size=get_batch_size('test'),
    max_seq_length=args.max_seq_length,
    num_workers=args.dataloader_workers,
    task=args.task
)
data_module.setup()

model = BioBERT(
    model_name_or_path=args.model_name_or_path,
    num_labels=num_labels[args.task],
    learning_rate=args.learning_rate,
    adam_epsilon=args.adam_epsilon,
    warmup_steps=args.warm_steps,
    weight_decay=args.weight_decay,
    train_batch_size=get_batch_size('train'),
    eval_batch_size=get_batch_size('valid')
)
trainer = Trainer(max_epochs=args.epochs,
                  gpus=args.gpus,
                  accelerator='ddp',
                  check_val_every_n_epoch=3,
                  )
trainer.fit(model, data_module)
trainer.test(ckpt_path='best')



