import os
import torch
import numpy as np
import argparse
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from qm9_dataset import QM9DGLDataset
from omegaconf import OmegaConf
from model import GNN_model
import torch.nn as nn
from tqdm import tqdm

"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def to_np(x):
    return x.cpu().detach().numpy()


def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, scheduler, FLAGS, device):
    model.train()
    num_iters = len(dataloader)
    for i, (g, y) in enumerate(dataloader):
        g = g.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # run model forward and compute loss
        pred = model(g)
        loss = loss_fnc(pred, y)

        # backprop
        loss.backward()
        optimizer.step()

        if i % FLAGS.train_params.print_epoch_interval == 0:
            print(f"[{epoch}|{i}] loss: {loss.item():.5f}")

        scheduler.step(epoch + i / num_iters)


def val_epoch(epoch, model, loss_fnc, dataloader, FLAGS, device):
    model.eval()
    total_loss = 0
    rescale_loss = 0
    for i, (g, y) in enumerate(tqdm(dataloader)):
        g = g.to(device)
        y = y.to(device)

        # run model forward and compute loss
        pred = model(g)
        loss = loss_fnc(pred, y)

        total_loss += loss.item()

    print(f"...[{epoch}|val] loss: {total_loss:.5f}")


def run_test(model, dataloader, device):
    model.eval()
    preds = []
    for g in tqdm(dataloader):
        g = g.to(device)
        pred = model(g)
        preds.append(to_np(pred))

    return np.concatenate(preds, axis=0)

# Loss function
def l1_loss(pred, target):
    loss = F.l1_loss(pred, target)
    return loss

################ 1. parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/gnn.yaml', help='configration for model')
parser.add_argument('--pretrained_path', type=str, default=None, help='configration for model')
args, extra_args = parser.parse_known_args()
FLAGS = OmegaConf.load(args.config)

# Create model directory
if not os.path.isdir(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

# Fix SEED
torch.manual_seed(FLAGS.train_params.seed)
np.random.seed(FLAGS.train_params.seed)

# Automatically choose GPU if available
device = gpu_setup(FLAGS['gpu']['use'], FLAGS['gpu']['id'])



################ 2. Prepare data
dataset = QM9DGLDataset(FLAGS.data_path,
                              FLAGS.task,
                              file_name='qm9_train_data.pt',
                              mode='train')

train_dataset, val_dataset = dataset.train_val_random_split(0.8)


train_loader = DataLoader(train_dataset,
                          batch_size=FLAGS.train_params.batch_size,
                          shuffle=True,
                          collate_fn=dataset.collate_fn,
                          num_workers=FLAGS.data.num_workers)

val_loader = DataLoader(val_dataset,
                        batch_size=FLAGS.train_params.batch_size,
                        shuffle=False,
                        collate_fn=dataset.collate_fn,
                        num_workers=FLAGS.data.num_workers)

# Test Dataset
test_dataset = QM9DGLDataset(FLAGS.data_path,
                         FLAGS.task,
                         file_name='qm9_test_data.pt',
                         mode='test')

test_loader = DataLoader(test_dataset,
                         batch_size=FLAGS.train_params.batch_size,
                         shuffle=False,
                         collate_fn=test_dataset.collate_fn,
                         num_workers=FLAGS.data.num_workers)

FLAGS.train_size = len(train_dataset)
FLAGS.val_size = len(val_dataset)
FLAGS.test_size = len(test_dataset)
print(f"Train set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

################ 2. Prepare model
model = GNN_model(FLAGS.graph_encoder_params)

if args.pretrained_path is not None:
    model.load_state_dict(torch.load(args.pretrained_path))

model.to(device)

criterion = l1_loss
optimizer = optim.Adam(model.parameters(), lr=FLAGS.train_params.init_lr)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                           FLAGS.train_params.epochs,
                                                           eta_min=FLAGS.train_params.min_lr)
                                                 

################ 2. Start training

# Run training
print('Begin training')
for epoch in range(FLAGS.train_params.epochs):
    train_epoch(epoch, model, criterion, train_loader, optimizer, scheduler, FLAGS, device)
    val_epoch(epoch, model, criterion, val_loader, FLAGS, device)
    
    # save checkpoint
    save_path = os.path.join(FLAGS.out_dir, f"{FLAGS.model}_{FLAGS.task}_{epoch}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint: {save_path}")


################ 3. Test
print('Begin test')
predictions = run_test(model, test_loader, device)
np.savetxt('pred.csv', predictions)
