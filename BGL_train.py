import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

from BGL_utils import (
    load_log_file, build_vocab, save_vocab, keys_to_ids,
    make_sequences, DeepLogModel, select_device,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLog training on BGL dataset')
    parser.add_argument('--num_layers',  default=2,      type=int)
    parser.add_argument('--hidden_size', default=64,     type=int)
    parser.add_argument('--window_size', default=10,     type=int)
    parser.add_argument('--num_epochs',  default=30,     type=int)
    parser.add_argument('--batch_size',  default=2*15,   type=int)
    parser.add_argument('--ratio',       default=1.0,    type=float,
                        help='Fraction of benign log to use for training (0~1], '
                             'taken sequentially from the start (time order)')
    parser.add_argument('-benign_log',  default='../Data/BGL/BGL_benign.log', type=str)
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU index to use (default: auto-select by free memory)')
    args = parser.parse_args()

    input_size = 1
    model_dir  = 'model'
    device     = select_device(args.gpu)
    use_cuda   = device.type == 'cuda'

    # ── Load & parse log ──────────────────────────────────────────────────────
    print(f'[BGL Train] log  : {args.benign_log}')
    print(f'[BGL Train] ratio: {args.ratio:.2f}')
    keys = load_log_file(args.benign_log, ratio=args.ratio)
    print(f'  log entries  : {len(keys):,}')

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab = build_vocab(keys)
    num_classes = len(vocab) + 1   # +1 for OOV (out-of-vocabulary) key
    print(f'  vocab size   : {len(vocab):,}  (num_classes={num_classes})')

    os.makedirs(model_dir, exist_ok=True)
    vocab_path = os.path.join(model_dir, f'bgl_vocab_r{args.ratio}.pkl')
    save_vocab(vocab, vocab_path)
    print(f'  vocab saved  : {vocab_path}')

    # ── Sequences ─────────────────────────────────────────────────────────────
    key_ids = keys_to_ids(keys, vocab)
    inputs, outputs = make_sequences(key_ids, args.window_size)
    print(f'  sequences    : {len(inputs):,}')

    dataset    = TensorDataset(torch.tensor(inputs, dtype=torch.float),
                               torch.tensor(outputs))
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, pin_memory=use_cuda,
                            num_workers=0)

    # ── Model & optimizer ─────────────────────────────────────────────────────
    log_tag = (f'BGL_Adam'
               f'_bs={args.batch_size}'
               f'_ep={args.num_epochs}'
               f'_ratio={args.ratio}')

    model     = DeepLogModel(input_size, args.hidden_size,
                             args.num_layers, num_classes).to(device)
    writer    = SummaryWriter(log_dir=os.path.join('log', log_tag))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f'\n[Training on {device}]')
    start_time = time.time()
    total_step = len(dataloader)

    for epoch in range(args.num_epochs):
        train_loss = 0.0
        for seq, label in dataloader:
            seq    = seq.view(-1, args.window_size, input_size).to(device, non_blocking=use_cuda)
            label  = label.to(device, non_blocking=use_cuda)
            output = model(seq)
            loss   = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / total_step
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], loss: {avg_loss:.4f}')
        writer.add_scalar('train_loss', avg_loss, epoch + 1)

    # ── Save ──────────────────────────────────────────────────────────────────
    elapsed    = time.time() - start_time
    model_path = os.path.join(model_dir, log_tag + '.pt')
    torch.save(model.state_dict(), model_path)
    writer.close()

    print(f'\nelapsed_time : {elapsed:.1f}s')
    print(f'model saved  : {model_path}')
    print('Finished Training')
