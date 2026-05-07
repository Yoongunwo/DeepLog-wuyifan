import re
import pickle
import torch
import torch.nn as nn


def select_device(gpu_id=None):
    """
    Select compute device and print selection info.
      gpu_id=None  → auto-select the GPU with the most free memory
      gpu_id=int   → use that specific GPU index
    Returns a torch.device.
    """
    if not torch.cuda.is_available():
        print('[Device] CPU (CUDA not available)')
        return torch.device('cpu')

    n = torch.cuda.device_count()
    if gpu_id is not None:
        if not (0 <= gpu_id < n):
            raise ValueError(f'GPU {gpu_id} not found (available: 0 ~ {n - 1})')
        chosen = gpu_id
    else:
        # Pick the GPU with the most free memory
        free_mem = [torch.cuda.mem_get_info(i)[0] for i in range(n)]
        chosen   = max(range(n), key=lambda i: free_mem[i])
        if n > 1:
            summary = '  |  '.join(
                f'GPU{i}: {free_mem[i] / 1024**3:.1f} GB free' for i in range(n)
            )
            print(f'[Device] Auto-selected GPU {chosen}  [{summary}]')

    dev = torch.device(f'cuda:{chosen}')
    torch.cuda.set_device(dev)
    torch.backends.cudnn.benchmark = True
    print(f'[Device] {torch.cuda.get_device_name(dev)}  (cuda:{chosen})')
    return dev


def extract_log_key(line):
    """
    BGL log format (space-separated, maxsplit=9):
      <label> <timestamp> <date> <location> <datetime> <node> <type> <component> <level> <content...>
    Returns a normalized template string as the log key.
    """
    parts = line.strip().split(None, 9)
    if len(parts) < 9:
        return None
    log_type  = parts[6]
    component = parts[7]
    level     = parts[8]
    content   = parts[9] if len(parts) > 9 else ""

    # Remove variable-length tokens
    content = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', content)
    content = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', content)
    content = re.sub(r':\d+\b', ':<*>', content)
    content = re.sub(r'\b\d+\b', '<*>', content)
    content = re.sub(r'\s+', ' ', content).strip()

    return f"{log_type}|{component}|{level}|{content}"


def load_log_file(log_path, ratio=1.0):
    """
    Read log file and return list of log-key strings.
    ratio: fraction of lines to use, taken sequentially from the start (time order).
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    n = max(1, int(len(lines) * ratio))
    lines = lines[:n]
    keys = []
    for line in lines:
        if line.strip():
            key = extract_log_key(line)
            if key:
                keys.append(key)
    return keys


def build_vocab(keys):
    """Build {log_key_string -> int_id} dict from a list of key strings."""
    unique_keys = sorted(set(keys))
    return {key: idx for idx, key in enumerate(unique_keys)}


def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def keys_to_ids(keys, vocab):
    """Map key strings to integer IDs; unknown keys map to len(vocab)."""
    unk_id = len(vocab)
    return [vocab.get(k, unk_id) for k in keys]


def make_sequences(key_ids, window_size):
    """Sliding-window sequence generation. Returns (inputs, outputs) lists."""
    inputs, outputs = [], []
    for i in range(len(key_ids) - window_size):
        inputs.append(key_ids[i:i + window_size])
        outputs.append(key_ids[i + window_size])
    return inputs, outputs


class DeepLogModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
