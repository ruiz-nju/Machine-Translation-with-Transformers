import os
import json
import torch
from torch.utils.data import Dataset

SPLIT = {"train": "training", "val": "validation", "test": "testing"}


class TranslationDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        assert split in SPLIT.keys(), "Invalid split name."
        split = SPLIT[split]
        self.data_dir = data_dir
        self.int2cn, self.int2en, self.cn2int, self.en2int = self._read_vocab()
        data_file = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found.")
        self.pairs = []  # [([...], [...]), (...), ...]
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                en, cn = line.strip().split("\t")  # 制表符作为分隔符
                # 处理英文中的@@符号
                en = en.replace("@@", "").split()  # 默认分隔符是空格
                cn = cn.split()
                self.pairs.append((en, cn))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en_tokens = [
            self.en2int.get(tok, self.en2int["<UNK>"]) for tok in self.pairs[idx][0]
        ]
        cn_tokens = [
            self.cn2int.get(tok, self.cn2int["<UNK>"]) for tok in self.pairs[idx][1]
        ]

        # 添加特殊标记
        en = [self.en2int["<BOS>"]] + en_tokens + [self.en2int["<EOS>"]]
        cn = [self.cn2int["<BOS>"]] + cn_tokens + [self.cn2int["<EOS>"]]

        return torch.LongTensor(en), torch.LongTensor(cn)

    def _read_vocab(self):
        data_dir = self.data_dir
        int2cn_file = os.path.join(data_dir, "int2word_cn.json")
        int2en_file = os.path.join(data_dir, "int2word_en.json")
        cn2int_file = os.path.join(data_dir, "word2int_cn.json")
        en2int_file = os.path.join(data_dir, "word2int_en.json")
        # 判断文件是否存在
        if (
            not os.path.exists(int2cn_file)
            or not os.path.exists(int2en_file)
            or not os.path.exists(cn2int_file)
            or not os.path.exists(en2int_file)
        ):
            raise FileNotFoundError(
                "Vocabulary files not found in the specified directory."
            )
        with open(int2cn_file, "r", encoding="utf-8") as f:
            int2cn = json.load(f)
        with open(int2en_file, "r", encoding="utf-8") as f:
            int2en = json.load(f)
        with open(cn2int_file, "r", encoding="utf-8") as f:
            cn2int = json.load(f)
        with open(en2int_file, "r", encoding="utf-8") as f:
            en2int = json.load(f)

        return int2cn, int2en, cn2int, en2int


if __name__ == "__main__":
    data_dir = "data/cmn-eng-simple"
    dataset = TranslationDataset(data_dir, split="train")
    print(f"Number of samples: {len(dataset)}\n")
    for i in range(3):
        en, cn = dataset[i]
        print("English: ", end=" ")
        for word in en:
            print(dataset.int2en[str(word.item())], end=" ")
        print("\nChinese: ", end=" ")
        for word in cn:
            print(dataset.int2cn[str(word.item())], end=" ")
        print("\n")
