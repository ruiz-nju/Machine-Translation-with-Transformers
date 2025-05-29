import os
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim import Adam
from model import Transformer
from utils.logger import setup_logger
from dataset import TranslationDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu


import config


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    en2int, cn2int = dataset.en2int, dataset.cn2int

    def _collate_fn(batch):
        en_batch, cn_batch = zip(*batch)

        # 加上 padding，补齐到相同长度，默认是在右侧进行 padding
        en_padded = pad_sequence(
            en_batch, batch_first=True, padding_value=en2int["<PAD>"]
        )

        cn_padded = pad_sequence(
            cn_batch, batch_first=True, padding_value=cn2int["<PAD>"]
        )
        return {
            "source": en_padded,
            "target": cn_padded,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


def convert_to_text(tensor, int2word):
    # 提取 <BOS> 和 <EOS> 中间的部分
    words = [int2word[str(i.item())] for i in tensor]
    if "<BOS>" in words and "<EOS>" in words:
        start_idx = words.index("<BOS>") + 1
        end_idx = words.index("<EOS>")
        text = " ".join(words[start_idx:end_idx])
    else:
        text = " ".join(words)
    return text


def save_plot(output_dir, train_losses, train_accs, val_losses, val_accs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Training Accuracy", marker="o")
    plt.plot(val_accs, label="Validation Accuracy", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()


def print_model_summary(model, depth=3):
    header = ["Layer (type)", "Output Shape", "Param #", "Trainable"]
    rows = []
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    # 递归遍历模型结构
    def _add_layer_info(module, name, depth):
        nonlocal total_params, trainable_params, non_trainable_params
        params = sum(np.prod(p.size()) for p in module.parameters())
        if params == 0:
            return

        # 参数统计
        trainable = any(p.requires_grad for p in module.parameters())
        total_params += params
        if trainable:
            trainable_params += params
        else:
            non_trainable_params += params

        # 构造输出形状（示例）
        output_shape = (
            "x".join(str(s) for s in module.example_output_shape)
            if hasattr(module, "example_output_shape")
            else "--"
        )

        # 添加到表格
        rows.append(
            [
                name + f" ({module.__class__.__name__})",
                f"[{output_shape}]",
                f"{params:,}",
                "Yes" if trainable else "No",
            ]
        )

        # 递归子模块
        if depth > 0:
            for child_name, child_module in module.named_children():
                _add_layer_info(child_module, f"{name}.{child_name}", depth - 1)

    # 遍历顶层模块
    for name, module in model.named_children():
        _add_layer_info(module, name, depth)

    # 打印表格
    from tabulate import tabulate

    print(tabulate(rows, headers=header, tablefmt="psql"))

    # 参数单位转换
    def _format_num(num):
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.1f}K"
        return str(num)

    # 打印汇总信息
    print(f"\n{'='*60}")
    print(f"Total params: {_format_num(total_params)} ({total_params:,})")
    print(f"Trainable params: {_format_num(trainable_params)} ({trainable_params:,})")
    print(
        f"Non-trainable params: {_format_num(non_trainable_params)} ({non_trainable_params:,})"
    )
    print(f"Model size: {total_params*4/(1024**2):.2f}MB (FP32)")  # 假设32位浮点
    print("=" * 60 + "\n")


def reset_config(args):
    if args.num_head:
        config.heads = args.num_head

    if args.num_layer:
        config.n_layers = args.num_layer

    config.output_dir = f"outputs/num_head_{config.heads}/num_layer_{config.n_layers}"


def main(args):
    reset_config(args)
    set_random_seed(config.seed)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    setup_logger(os.path.join(config.output_dir, "logs"))

    period = args.period  # train or eval

    train_set = TranslationDataset(config.data_dir, split="train")
    val_set = TranslationDataset(config.data_dir, split="val")
    test_set = TranslationDataset(config.data_dir, split="test")
    train_loader = build_dataloader(train_set, batch_size=config.batch_size)
    val_loader = build_dataloader(val_set, batch_size=config.batch_size)
    test_loader = build_dataloader(test_set, batch_size=config.batch_size)
    en2int, cn2int, int2en, int2cn = (
        train_set.en2int,
        train_set.cn2int,
        train_set.int2en,
        train_set.int2cn,
    )
    en_vocab_size = len(en2int)
    cn_vocab_size = len(cn2int)
    PAD_ID = en2int["<PAD>"]
    BOS_ID = cn2int["<BOS>"]

    model = Transformer(
        src_vocab_size=en_vocab_size,
        dst_vocab_size=cn_vocab_size,
        pad_idx=PAD_ID,
        d_model=config.d_model,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        heads=config.heads,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    ).to(config.device)
    print_model_summary(model)
    if period == "train":
        optimizer = Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        train_losses = []
        train_accs = []
        val_losses = []  # 验证损失列表
        val_accs = []  # 验证准确率列表

        for epoch in range(config.n_epochs):
            model.train()
            epoch_total_loss = 0.0
            epoch_total_correct = 0.0
            epoch_total_non_pad = 0.0

            count = 1
            total = len(train_loader)

            tic = time.time()
            for i, batch in enumerate(train_loader):
                x = torch.LongTensor(batch["source"]).to(
                    config.device
                )  # torch.Size([32, 19])
                y = torch.LongTensor(batch["target"]).to(
                    config.device
                )  # torch.Size([32, 17])
                # 由于 Transformer 是在用前 i 个 token 预测第 i+1 个 token
                # 考虑并行计算的话，我们可以直接输入前 n-1 个 token，并行预测后 n-1 个 token
                y_output = y[:, :-1]
                y_label = y[:, 1:]
                y_hat = model(x, y_output)
                y_label_mask = y_label != PAD_ID
                preds = torch.argmax(y_hat, -1)

                correct = preds == y_label
                acc = torch.sum(y_label_mask * correct) / torch.sum(y_label_mask)

                n, seq_len = y_label.shape
                y_hat = torch.reshape(y_hat, (n * seq_len, -1))
                y_label = torch.reshape(y_label, (n * seq_len,))
                loss = criterion(y_hat, y_label)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                epoch_total_loss += loss.item()
                current_correct = torch.sum(y_label_mask * correct).item()
                current_non_pad = torch.sum(y_label_mask).item()
                epoch_total_correct += current_correct
                epoch_total_non_pad += current_non_pad

                if count % config.print_interval == 0 or count == total:
                    toc = time.time()
                    interval = toc - tic
                    minutes = int(interval // 60)
                    seconds = int(interval % 60)
                    print(
                        f"Epoch: [{epoch+1}/{config.n_epochs}], Batch: [{count}/{total}], "
                        f"Loss: {loss.item()}, Acc: {acc.item()}, Time: {minutes:02d}:{seconds:02d}"
                    )
                count += 1
            avg_epoch_loss = epoch_total_loss / total
            avg_epoch_acc = epoch_total_correct / epoch_total_non_pad
            train_losses.append(avg_epoch_loss)
            train_accs.append(avg_epoch_acc)

            # 计算验证集准确度
            model.eval()
            val_total_loss = 0.0
            val_total_correct = 0.0
            val_total_non_pad = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = torch.LongTensor(batch["source"]).to(config.device)
                    y = torch.LongTensor(batch["target"]).to(config.device)
                    y_output = y[:, :-1]
                    y_label = y[:, 1:]

                    # 前向传播
                    y_hat = model(x, y_output)
                    y_label_mask = y_label != PAD_ID
                    preds = torch.argmax(y_hat, -1)

                    # 计算准确率
                    correct = preds == y_label
                    current_correct = torch.sum(y_label_mask * correct).item()
                    current_non_pad = torch.sum(y_label_mask).item()
                    val_total_correct += current_correct
                    val_total_non_pad += current_non_pad

                    # 计算损失
                    n, seq_len = y_label.shape
                    y_hat_flat = torch.reshape(y_hat, (n * seq_len, -1))
                    y_label_flat = torch.reshape(y_label, (n * seq_len,))
                    loss = criterion(y_hat_flat, y_label_flat)
                    val_total_loss += loss.item()

            # 计算验证集平均指标
            avg_val_loss = val_total_loss / len(val_loader)
            avg_val_acc = val_total_correct / val_total_non_pad
            val_losses.append(avg_val_loss)
            val_accs.append(avg_val_acc)

            # 打印训练和验证指标
            print(
                f"Epoch: [{epoch+1}/{config.n_epochs}], "
                f"Avg Val loss: {avg_val_loss:.4f}, Avg Val acc: {avg_val_acc:.4f}"
            )
            print("=" * 100)

        model_path = os.path.join(config.output_dir, "final_model.pth")
        torch.save(model.state_dict(), model_path)
        save_plot(config.output_dir, train_losses, train_accs, val_losses, val_accs)
        print("Training completed.")

    elif period == "eval":
        model_path = os.path.join(config.output_dir, "final_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        with torch.no_grad():
            en_origin = []
            cn_standard = []
            cn_output = []
            for batch in tqdm(test_loader):
                x = torch.LongTensor(batch["source"]).to(config.device)
                y = torch.LongTensor(batch["target"]).to(config.device)
                batch_size = x.shape[0]
                max_len = y.shape[1]
                y_output = torch.full(
                    (batch_size, max_len),
                    PAD_ID,
                    dtype=torch.long,
                    device=config.device,
                )
                y_output[:, 0] = BOS_ID
                for cur_idx in range(1, max_len):
                    decoder_input = y_output[:, :cur_idx]
                    output = model(x, decoder_input)
                    next_tokens = torch.argmax(output[:, -1, :], dim=-1)
                    y_output[:, cur_idx] = next_tokens
                for j in range(batch_size):
                    en_origin.append(convert_to_text(x[j], int2en))
                    cn_standard.append(convert_to_text(y[j], int2cn))
                    cn_output.append(convert_to_text(y_output[j], int2cn))
        references = [[ref.split()] for ref in cn_standard]
        hypotheses = [hyp.split() for hyp in cn_output]
        bleu_score = corpus_bleu(references, hypotheses)
        print(f"BLEU Score: {bleu_score:.4f}")

        print("-" * 50)
        for i in range(3):
            print(f"Original: {en_origin[i]}")
            print(f"Standard: {cn_standard[i]}")
            print(f"Translated: {cn_output[i]}")
            print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--period",
        type=str,
        default="train",
        choices=["train", "eval"],
    )
    parser.add_argument(
        "--num_head",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=6,
    )
    args = parser.parse_args()
    main(args)
