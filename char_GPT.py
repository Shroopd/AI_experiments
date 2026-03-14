import torch
import numpy
import pickle
import uuid

import my_experiments as mxp
import torch.nn as nn
import torch.nn.functional as ff

from torch import Tensor


def mask(x: Tensor) -> Tensor:
    return x.movedim(-1, 0).triu(diagonal=0).movedim(0, -1)


class MLP(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.all = nn.Sequential(
            # mxp.LinearActivateZP(dims, dims * 2, nn.SiLU()),
            nn.Linear(dims, dims * 2, False),
            mxp.Swishmoid(dims * 2, 1),
            nn.Linear(2 * dims, dims, False),
            # nn.Tanh(),
            # nn.Linear(dims, False),
        )

    def forward(self, X):
        return self.all(X)


class GPT(nn.Module):
    def __init__(self, dims: int, layers: int, metadata) -> None:
        super().__init__()
        self.vocab_size = metadata["vocab_size"]

        self.pos_encode = mxp.PosEncode(dims // 2, 2, 2048)
        self.encode = nn.Bilinear(vocab_size, dims, dims)

        # with torch.no_grad():
        #     self.encode.weight *= 0.0001
        # self.encode = nn.Parameter(torch.randn(self.vocab_size, dims) / layers)

        self.all = nn.Sequential(
            *(mxp.FractalTransformer(dims, 2, MLP, mask=mask) for _ in range(layers))
        )

        self.decode = nn.Linear(dims, self.vocab_size, False)

    def forward(self, X: Tensor) -> Tensor:
        pos_encoding = self.pos_encode.forward(
            len(X) - torch.arange(X.shape[-1], device="cuda")
        ).cuda()
        X = ff.one_hot(X, self.vocab_size).float()
        pos_encoding = pos_encoding.unsqueeze(0).expand(X.shape[0], -1, -1)
        # print(X.shape, pos_encoding.shape)
        X = self.encode(X, pos_encoding)
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
        floats = floats[0, :, :]
    return int_to_str(torch.multinomial(torch.softmax(floats * temp, -1), 1))


model = GPT(256, 4, meta)
# model = nn.Linear(vocab_size, vocab_size, bias=False)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = nn.CrossEntropyLoss()

# model.load_state_dict(
#     torch.load("checkpoint/model_be3e608d.pt", weights_only=True), strict=False
# )

model.cuda()

# with torch.autograd.set_detect_anomaly(True):
# context_size = 16
# epochs = 10
# batches = 1280


def train(epochs, context_size, batches):
    for e in range(epochs):
        train_set_input_list = []
        train_set_target_list = []
        for i in range(
            int(torch.randint(0, context_size, (1,)).item()),
            len(train_data),
            context_size,
        ):
            context_block = torch.tensor(
                train_data[i : i + context_size],
                dtype=torch.long,
                device="cuda",
            )
            target = torch.tensor(
                train_data[i + 1 : i + context_size + 1],
                dtype=torch.long,
                device="cuda",
            )
            if len(target) == context_size:
                train_set_input_list.append(context_block)
                train_set_target_list.append(target)
            del context_block, target

        train_set_input = torch.stack(train_set_input_list)
        train_set_target = torch.stack(train_set_target_list)

        for b in range(0, batches):
            input_data = train_set_input[b::batches]
            target = train_set_target[b::batches]
            optimizer.zero_grad()

            # target = train_data[i + 1 : i + context_size + 1]

            # context_block = ff.one_hot(context_block, vocab_size).float()

            result = model(input_data)
            if b == 0:
                print(result.shape)
                print(target.shape)

            result_dist = ff.softmax(result[0], dim=-1)
            if b == 0:
                print(result_dist.shape)
            result_select = torch.multinomial(result_dist, 1)

            if b == 0:
                # print(target[0])
                print(int_to_str(target[0]))
                print("\n------------------\n")
                # print(result_dist[0])
                print(int_to_str(result_select))

            loss = loss_func(result.mT, target)

            # if i == 0:

            print(b, loss.item())
            # if loss.item() > 100:
            #     raise RuntimeError

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        file_name = "model_" + uuid.uuid4().hex[:8]
        print(e, file_name)
        torch.save(
            model.state_dict(),
            "checkpoint/" + file_name + ".pt",
        )
        torch.cuda.empty_cache()


def test(temp, context_size, offset=0):
    # start = input("::Testing Start::\n")
    start = int_to_str(torch.tensor(eval_data[offset : offset + context_size]).long())
    text = start
    print("\n::Start::\n" + text)
    for i in range(context_size):
        context = text[-min(context_size, len(text)) :]
        out = model(str_to_int(context).cuda())
        new_chars = float_to_str(out, temp)
        # print("::::" + new_chars + "::::")
        # break
        text = text + new_chars[-1]
        # print(new_char, end="")
    print("\n::Result::\n" + text)


for i in range(32):
    train(i, int(16 * 16 ** (i / 32)), int(64 * 16 ** (i / 32)))
# train(1,20,512)
# test(1, 128, 512)
