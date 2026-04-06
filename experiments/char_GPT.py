import torch
import numpy
import pickle
import uuid
import random

import my_experiments as mxp
import torch.nn as nn
import torch.nn.functional as ff

from torch import Tensor
from my_experiments import mask2d


class MLP(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.all = nn.Sequential(
            nn.Linear(dims, dims * 2, False),
            mxp.Silulog(),
            nn.Linear(2 * dims, dims, False),
        )

    def forward(self, X):
        return self.all(X)


class GPT(nn.Module):
    def __init__(self, dims: int, layers: int, metadata) -> None:
        super().__init__()
        self.vocab_size = metadata["vocab_size"]

        self.pos_encode = mxp.RotPosEncode(dims, 1, 2, 2048)
        self.encode = nn.Linear(vocab_size, dims)

        # with torch.no_grad():
        #     self.encode.weight *= 0.0001
        # self.encode = nn.Parameter(torch.randn(self.vocab_size, dims) / layers)

        self.all = nn.Sequential(
            *(
                mxp.FractalTransformer(
                    dims, 2, MLP, mask=mask2d, pos_encoder=self.pos_encode
                )
                for _ in range(layers)
            )
        )

        self.decode = nn.Linear(dims, self.vocab_size, False)

    def forward(self, X: Tensor) -> Tensor:
        # pos_encoding = self.pos_encode.forward(
        #     len(X) - torch.arange(X.shape[-1], device="cuda")
        # ).cuda()
        X = ff.one_hot(X, self.vocab_size).float()
        # pos_encoding = pos_encoding.unsqueeze(0).expand(X.shape[0], -1, -1)
        # print(X.shape, pos_encoding.shape)
        X = self.encode(X)
        for module in self.all:
            X = X + module(X)
        X = self.decode(X)
        return X


with open("data/shakespeare/meta.pkl", "rb") as file:
    meta = pickle.load(file)

"""
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
"""

vocab_size = meta["vocab_size"]

train_data = numpy.fromfile("data/shakespeare/train.bin", dtype="uint16")
eval_data = numpy.fromfile("data/shakespeare/val.bin", dtype="uint16")

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens


def str_to_int(string: str):
    foo = meta["stoi"]
    bar = torch.zeros(len(string))
    for i in range(len(string)):
        bar[i] = foo[string[i]]
    return bar.long().unsqueeze(-2)


def int_to_str(integers: Tensor):
    foo = meta["itos"]
    return "".join(foo[integers[i].item()] for i in range(len(integers)))


def float_to_str(floats: Tensor, temp):
    if len(floats.shape) == 3:
        floats = floats[0]
    return int_to_str(torch.multinomial(torch.softmax(floats / temp, dim=-1), 1))


model = GPT(512, 6, meta)
# model = nn.Linear(vocab_size, vocab_size, bias=False)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# loss_func = nn.CrossEntropyLoss()

# model.load_state_dict(
#     torch.load("checkpoint/model_774bc79e_1.pt", weights_only=True), strict=False
# )

model.cuda()

# with torch.autograd.set_detect_anomaly(True):
# context_size = 16
# epochs = 10
# batches = 1280


def save(torch_model, num):
    file_name = "model_" + uuid.uuid4().hex[:8] + "_" + str(num)
    print(file_name)
    torch.save(
        torch_model.state_dict(),
        "checkpoint/" + file_name + ".pt",
    )


def train(epochs, context_size, batches):
    print(epochs, context_size, batches)
    for e in range(epochs):
        train_set_list = []
        # train_set_input_list = []
        # train_set_target_list = []
        for i in range(
            int(torch.randint(0, context_size, (1,)).item()),
            len(train_data),
            context_size + 1,
        ):
            data_block = torch.tensor(
                train_data[i : i + context_size + 1],
                dtype=torch.long,
                device="cuda",
            )
            if len(data_block) == context_size + 1:
                train_set_list.append(data_block)
        random.shuffle(train_set_list)
        train_set = torch.stack(train_set_list)
        # train_set_input = torch.stack(train_set_input_list)
        # train_set_target = torch.stack(train_set_target_list)

        for b in range(0, batches):
            input_data = train_set[b::batches, :-1]
            target = train_set[b::batches, 1:]
            optimizer.zero_grad()

            # target = train_data[i + 1 : i + context_size + 1]

            # context_block = ff.one_hot(context_block, vocab_size).float()

            result = model(input_data)

            if b == 0:
                print("input: ", input_data.shape)
                print("target:", target.shape)
                print("result:", result.shape)

            # result_dist = ff.softmax(result[0], dim=-1)
            # if b == 0:
            #     print(result_dist.shape)
            # result_select = torch.multinomial(result_dist, 1)

            if b == 0:
                # print(target[0])
                print(">>" + int_to_str(target[0]) + "<<")
                print("\n------------------\n")
                # print(result_dist[0])
                # print(">>" + float_to_str(result, 0.5) + "<<")
                print(">>" + float_to_str(result, 1.0) + "<<")
                # print(">>" + float_to_str(result, 2.0) + "<<")

            loss = ff.cross_entropy(result.mT, target)

            # if i == 0:

            # print(b, loss.item())
            if b % (batches // 10) == 0:
                print(b, loss.item())
            # if loss.item() > 100:
            #     raise RuntimeError

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        save(model, e)
        torch.cuda.empty_cache()


def test(temp, context_size, length, offset=0):
    with torch.no_grad():
        # start = input("::Testing Start::\n")
        start = int_to_str(
            torch.tensor(train_data[offset : offset + context_size]).long()
        )
        text = start
        print("\n>>Start<<\n" + ">>" + text + "<<")
        for i in range(length):
            context = text[:]
            out = model(str_to_int(context).cuda())
            new_chars = float_to_str(out[-1:, -1], temp)
            # print("::::" + new_chars + "::::")
            # break
            text = text + new_chars[-1]
            # print(new_char, end="")
        print("\n>>Result<<\n" + ">>" + text + "<<")


# train(16, 64, 1000)
# train(32, 128, 1024)
# for i in range(32):
#     train(1, int(32 * 8 ** (i / 32)), int(256 * 4 ** (i / 32)))
# train(1,20,512)

# test(1, 14, 8,200)

# for i in range(-16, 8 + 1):
#     temp = 2 ** (i / 8)
#     test(temp, 192, 64, 320)
#     print("i:", i)
#     print("temp:", temp)
