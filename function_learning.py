import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import RMSprop

from models import MLP, NAC, NALU

NORMALIZE = True
NUM_LAYERS = 4
HIDDEN_DIM = 4
LEARNING_RATE = 1e-2
NUM_ITERS = int(2e5)
RANGE = [0, 10]
ARITHMETIC_FUNCTIONS = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'squared': lambda x, y: torch.pow(x, 2),
    'root': lambda x, y: torch.sqrt(x),
    'log': lambda x, y: torch.log(x)
}


def generate_data(num_train, num_test, dim, num_sum, fn, support):
    data = torch.zeros([dim, 1], dtype=torch.float).uniform_(*support)
    X, y = [], []
    for i in range(num_train + num_test):
        idx_a = random.sample(range(dim), num_sum)
        idx_b = random.sample([x for x in range(dim) if x not in idx_a], num_sum)
        a, b = data[idx_a].sum(), data[idx_b].sum()
        X.append([a, b])
        y.append(fn(a, b))
    X = torch.tensor(X)
    y = torch.tensor(y).unsqueeze_(1)
    indices = list(range(num_train + num_test))
    np.random.shuffle(indices)
    X_train, y_train = X[indices[num_test:]], y[indices[num_test:]]
    X_test, y_test = X[indices[:num_test]], y[indices[:num_test]]
    return X_train, y_train, X_test, y_test


def train(model, optimizer, data, target, num_iters):
    losses = []
    meandiffs = []
    for i in range(1, num_iters + 1):
        out = model(data)
        loss = F.mse_loss(out, target)
        mea = torch.mean(torch.abs(target - out))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        meandiffs.append(mea.item())
        print(f'\r epoch: [{i+1}/{num_iters}], loss: {loss.item()}, mean_diff: {mea.item()}', end='')
    return losses, meandiffs


def test(model, data, target):
    with torch.no_grad():
        out = model(data)
        return torch.abs(target - out)


def main():
    models = [
        NALU(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1
        ),
    ]

    results = {}
    for fn_str, fn in ARITHMETIC_FUNCTIONS.items():
        print('[*] Testing function: {}'.format(fn_str))
        results[fn_str] = []

        # dataset
        X_train, y_train, X_test, y_test = generate_data(
            num_train=500, num_test=50,
            dim=100, num_sum=5, fn=fn,
            support=RANGE,
        )

        # random model
        random_mse = []
        for i in range(100):
            net = MLP(
                num_layers=NUM_LAYERS, in_dim=2,
                hidden_dim=HIDDEN_DIM, out_dim=1,
                activation='relu6',
            )
            mse = test(net, X_test, y_test)
            random_mse.append(mse.mean().item())
        results[fn_str].append(np.mean(random_mse))

        # others
        for net in models:
            print("\tTraining {}...".format(net.__str__().split("(")[0]))
            optim = RMSprop(net.parameters(), lr=LEARNING_RATE)
            train(net, optim, X_train, y_train, NUM_ITERS)
            mse = test(net, X_test, y_test).mean().item()
            print("\t\tTest finished {}".format(mse))
            results[fn_str].append(mse)

        print("\n---------------RESULTS------------------")

        print("Operation\tNALU")
        for k, v in results.items():
            print("{}\t".format(k), end='')
            rand = results[k][0]
            mses = [100.0 * x / rand for x in results[k][1:]]
            if NORMALIZE:
                for ms in mses:
                    print("{:.3f}\t".format(ms), end='')
                print()
            else:
                for ms in results[k][1:]:
                    print("{:.3f}\t".format(ms), end='')
                print()


if __name__ == '__main__':
    main()
