import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_baseline import make_model
from train_utils import NoamOpt, LabelSmoothing
import matplotlib.pyplot as plt
import csv
import os
import random
import numpy as np

# --- 黄金法则：固定随机种子 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

class SimpleVocab:
    def __init__(self):
        self.stoi = {'<pad>': PAD_IDX, '<bos>': BOS_IDX, '<eos>': EOS_IDX, '<unk>': UNK_IDX}
        self.itos = {PAD_IDX: '<pad>', BOS_IDX: '<bos>', EOS_IDX: '<eos>', UNK_IDX: '<unk>'}
        self.vocab_size = 4
    def add_sentence(self, sentence):
        for word in sentence.lower().split():
            if word not in self.stoi:
                self.stoi[word] = self.vocab_size
                self.itos[self.vocab_size] = word
                self.vocab_size += 1
    def encode(self, sentence):
        return [self.stoi.get(word, UNK_IDX) for word in sentence.lower().split()]

def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0

def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask

def collate_batch(batch, src_vocab, tgt_vocab, max_len=50):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_tokens = [BOS_IDX] + src_vocab.encode(item['de']) + [EOS_IDX]
        tgt_tokens = [BOS_IDX] + tgt_vocab.encode(item['en']) + [EOS_IDX]
        src_tokens = src_tokens[:max_len] + [PAD_IDX] * max(0, max_len - len(src_tokens))
        tgt_tokens = tgt_tokens[:max_len] + [PAD_IDX] * max(0, max_len - len(tgt_tokens))
        src_batch.append(src_tokens)
        tgt_batch.append(tgt_tokens)
    src = torch.tensor(src_batch, dtype=torch.long)
    tgt = torch.tensor(tgt_batch, dtype=torch.long)
    tgt_in, tgt_y = tgt[:, :-1], tgt[:, 1:]
    src_mask = (src != PAD_IDX).unsqueeze(-2)
    tgt_mask = make_std_mask(tgt_in, PAD_IDX)
    ntokens = (tgt_y != PAD_IDX).data.sum()
    return src, tgt_in, tgt_y, src_mask, tgt_mask, ntokens

def run_epoch(data_loader, model, criterion, optimizer, device):
    model.train()
    total_loss, total_tokens = 0.0, 0
    steps, losses = [], []
    for i, (src, tgt_in, tgt_y, src_mask, tgt_mask, ntokens) in enumerate(data_loader):
        src, tgt_in, tgt_y = src.to(device), tgt_in.to(device), tgt_y.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        out = model.generator(model(src, tgt_in, src_mask, tgt_mask))
        loss = criterion(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / ntokens
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * ntokens.item()
        total_tokens += ntokens.item()
        if i % 10 == 0:
            print(f"Step {i:4d} | Loss: {loss.item():.4f}")
            steps.append(i); losses.append(loss.item())
    return total_loss / total_tokens, steps, losses

def save_experiment_data(exp_name, avg_loss):
    file_exists = os.path.isfile('experiment_results.csv')
    with open('experiment_results.csv', mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Experiment Name', 'Average Loss'])
        writer.writerow([exp_name, f"{avg_loss:.4f}"])

def plot_and_save_loss(steps, losses, exp_name):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title(f'Training Loss Curve - {exp_name}', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Loss (MAE/CrossEntropy)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{exp_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # --- 1. 初始化与设置参数 ---
    set_seed(42) # 固定种子，确保每次分词、打乱顺序完全一致
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXPERIMENT_NAME = "Exp_2.1_Learnable_PE"
    
    print(f"当前实验: {EXPERIMENT_NAME} | 设备: {device}")
    dataset = load_dataset("bentrevett/multi30k", split="train[:5000]") 
    src_vocab, tgt_vocab = SimpleVocab(), SimpleVocab()
    for item in dataset:
        src_vocab.add_sentence(item['de'])
        tgt_vocab.add_sentence(item['en'])
        
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, 
        collate_fn=lambda b: collate_batch(b, src_vocab, tgt_vocab)
    )
    
    model = make_model(src_vocab.vocab_size, tgt_vocab.vocab_size, N=2, d_model=256, d_ff=1024, h=8).to(device)
    criterion = LabelSmoothing(size=tgt_vocab.vocab_size, padding_idx=PAD_IDX, smoothing=0.1).to(device)
    optimizer = NoamOpt(model_size=256, factor=1, warmup=400,
                        optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    
    # --- 2. 执行训练 ---
    avg_loss, steps, losses = run_epoch(dataloader, model, criterion, optimizer, device)
    
    # --- 3. 保存结果供论文撰写使用 ---
    save_experiment_data(EXPERIMENT_NAME, avg_loss)
    plot_and_save_loss(steps, losses, EXPERIMENT_NAME)
    
    print("="*40)
    print(f"[{EXPERIMENT_NAME}] 训练完成！平均 Loss: {avg_loss:.4f}")
    print("数据已自动写入 experiment_results.csv 并在当前目录生成了 Loss 曲线图。")