import torch

seed = 2025
data_dir = "data/cmn-eng-simple"
d_model = 512  # model dimension
d_ff = 2048  # feed forward dimension
n_layers = 6  # encoder and decoder layers
heads = 8  # number of heads
dropout = 0.1  # dropout rate
max_seq_len = 100  # max sequence length
batch_size = 16  # batch size
lr = 1e-4  # learning rate
n_epochs = 10  # number of epochs
print_interval = 50  # print interval
output_dir = f"outputs"
device = "cuda" if torch.cuda.is_available() else "cpu"  # device
