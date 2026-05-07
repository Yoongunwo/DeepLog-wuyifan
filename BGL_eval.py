import time
import torch
import argparse

from BGL_utils import (
    load_log_file, load_vocab, keys_to_ids,
    make_sequences, DeepLogModel, select_device,
)


def evaluate(model, inputs, outputs, window_size, num_candidates, device,
             batch_size=4096, input_size=1):
    """
    Batch-inference evaluation using the top-g candidate approach.
    A window is flagged as anomaly when the actual next key is NOT
    among the top-g predicted keys.
    Returns (num_detected, total_windows).
    """
    use_cuda = device.type == 'cuda'
    model.eval()
    detected = 0
    n = len(inputs)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            b_in  = inputs[i:i + batch_size]
            b_out = outputs[i:i + batch_size]

            seq    = torch.tensor(b_in, dtype=torch.float) \
                         .view(-1, window_size, input_size).to(device, non_blocking=use_cuda)
            labels = torch.tensor(b_out).to(device, non_blocking=use_cuda)  # (B,)

            out   = model(seq)                                               # (B, num_keys)
            top_g = torch.argsort(out, dim=1, descending=True)[:, :num_candidates]  # (B, g)

            match     = (top_g == labels.unsqueeze(1)).any(dim=1)           # (B,)
            detected += (~match).sum().item()

    return detected, n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLog evaluation on BGL dataset')
    parser.add_argument('-num_layers',     default=2,    type=int)
    parser.add_argument('-hidden_size',    default=64,   type=int)
    parser.add_argument('-window_size',    default=10,   type=int)
    parser.add_argument('-num_candidates', default=9,    type=int,
                        help='Top-g candidates for anomaly detection')
    parser.add_argument('-model_path',     required=True, type=str,
                        help='Path to trained .pt model file')
    parser.add_argument('-vocab_path',     default='model/bgl_vocab.pkl', type=str)
    parser.add_argument('-benign_log',     default='../Data/BGL/BGL_benign.log', type=str)
    parser.add_argument('-anomaly_log',    default='../Data/BGL/BGL_anomaly.log', type=str)
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU index to use (default: auto-select by free memory)')
    args = parser.parse_args()

    input_size = 1
    device     = select_device(args.gpu)
    use_cuda   = device.type == 'cuda'

    # ── Load vocab & model ────────────────────────────────────────────────────
    vocab = load_vocab(args.vocab_path)
    num_classes = len(vocab) + 1
    print(f'Vocab loaded : {len(vocab):,} keys  (num_classes={num_classes})')

    model = DeepLogModel(input_size, args.hidden_size,
                         args.num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f'Model loaded : {args.model_path}')

    start = time.time()

    # ── Benign (normal) evaluation ────────────────────────────────────────────
    print(f'\nEvaluating benign  : {args.benign_log}')
    benign_keys = load_log_file(args.benign_log)
    benign_ids  = keys_to_ids(benign_keys, vocab)
    b_in, b_out = make_sequences(benign_ids, args.window_size)
    fp, total_n = evaluate(model, b_in, b_out, args.window_size, args.num_candidates, device)
    tn = total_n - fp

    # ── Anomaly evaluation ────────────────────────────────────────────────────
    print(f'Evaluating anomaly : {args.anomaly_log}')
    anomaly_keys = load_log_file(args.anomaly_log)
    anomaly_ids  = keys_to_ids(anomaly_keys, vocab)
    a_in, a_out  = make_sequences(anomaly_ids, args.window_size)
    tp, total_a  = evaluate(model, a_in, a_out, args.window_size, args.num_candidates, device)
    fn = total_a - tp

    # ── Metrics ───────────────────────────────────────────────────────────────
    elapsed   = time.time() - start
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\n{'=' * 56}")
    print(f"  Results  (num_candidates={args.num_candidates})")
    print(f"{'=' * 56}")
    print(f"  Benign  windows : {total_n:>10,}  |  FP={fp:,}  TN={tn:,}")
    print(f"  Anomaly windows : {total_a:>10,}  |  TP={tp:,}  FN={fn:,}")
    print(f"{'─' * 56}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Elapsed   : {elapsed:.1f}s")
    print('Finished Predicting')
