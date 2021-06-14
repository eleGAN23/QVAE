import torch
import random
import numpy as np
from scipy.signal import lfilter


### Build dataset for toy example #1
def QProper_signals1(len_train, len_val, len_test, n):

    # Generate Q-Proper signal for train
    train_dataset = torch.zeros((len_train, n * 4))
    for i in range(len_train):
        q0 = torch.randn(n) + 0.03
        q1 = torch.randn(n) + 0.03
        q2 = torch.randn(n) + 0.03
        q3 = torch.randn(n) + 0.03
        q = torch.cat([q0, q1, q2, q3], dim=0)
        train_dataset[i, :] = q
        train_dataset / torch.max(train_dataset)

    # Build validation set
    val_dataset = torch.zeros((len_val, n * 4))
    for i in range(len_val):
        q0 = torch.randn(n) + 0.03
        q1 = torch.randn(n) + 0.03
        q2 = torch.randn(n) + 0.03
        q3 = torch.randn(n) + 0.03
        q = torch.cat([q0, q1, q2, q3], dim=0)
        val_dataset[i, :] = q
        val_dataset / torch.max(val_dataset)

    # Build test set
    test_dataset = torch.zeros((len_test, n * 4))
    for i in range(len_test):
        q0 = torch.randn(n) + 0.03
        q1 = torch.randn(n) + 0.03
        q2 = torch.randn(n) + 0.03
        q3 = torch.randn(n) + 0.03
        q = torch.cat([q0, q1, q2, q3], dim=0)
        test_dataset[i, :] = q
        test_dataset / torch.max(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset)
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset)
    )

    return train_loader, val_loader, test_loader

### Build dataset for toy example #1
def QProper_signals2(len_train, len_val, len_test, n):

    # Generate Q-Proper signal for train
    b = 0.5
    train_dataset = torch.FloatTensor(
        lfilter([np.sqrt(1-b**2)], [1 -b], (1/(np.sqrt(4)))*(np.random.randn(len_train,4)))
    )

    # Build validation set
    val_dataset = torch.FloatTensor(
        lfilter([np.sqrt(1-b**2)], [1 -b], (1/(np.sqrt(4)))*(np.random.randn(len_val,4)))
    )

    # Build test set
    test_dataset = torch.FloatTensor(
        lfilter([np.sqrt(1-b**2)], [1 -b], (1/(np.sqrt(4)))*(np.random.randn(len_test,4)))
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset)
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset)
    )

    return train_loader, val_loader, test_loader

def QImproper_signals1(len_train, len_val, len_test, n):
    
    w = torch.randn(len_train, 1) + 0.03
    q0 = torch.FloatTensor([0]).view(1, 1)
    q1 = torch.randn(1).view(1, 1)
    q2 = torch.randn(1).view(1, 1)
    q3 = torch.randn(1).view(1, 1)
    c = torch.cat([q0, q1, q2, q3], dim=1)
    a = torch.exp(c)
    q = a*w
    # q /= torch.max(q)
    train_dataset = q

    w = torch.randn(len_val, 1) + 0.03
    q0 = torch.FloatTensor([0]).view(1, 1)
    q1 = torch.randn(1).view(1, 1)
    q2 = torch.randn(1).view(1, 1)
    q3 = torch.randn(1).view(1, 1)
    c = torch.cat([q0, q1, q2, q3], dim=1)
    a = torch.exp(c)
    q = a*w
    # q /= torch.max(q)
    val_dataset = q

    w = torch.randn(len_test, 1) + 0.03
    q0 = torch.FloatTensor([0]).view(1, 1)
    q1 = torch.randn(1).view(1, 1)
    q2 = torch.randn(1).view(1, 1)
    q3 = torch.randn(1).view(1, 1)
    c = torch.cat([q0, q1, q2, q3], dim=1)
    a = torch.exp(c)
    q = a*w
    # q /= torch.max(q)
    test_dataset = q

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset)
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset)
    )

    return train_loader, val_loader, test_loader

def QImproper_signals2(len_train, len_val, len_test, n):

    w = np.random.randn(len_train, 1) + 0.03
    w = w / np.max(w)
    b = [0.75, 0.78, 0.82, 0.85]
    q0 = torch.FloatTensor(lfilter([np.sqrt(1 - b[0] ** 2)], [1 - b[0]], w))
    q1 = torch.FloatTensor(lfilter([np.sqrt(1 - b[1] ** 2)], [1 - b[1]], w))
    q2 = torch.FloatTensor(lfilter([np.sqrt(1 - b[2] ** 2)], [1 - b[2]], w))
    q3 = torch.FloatTensor(lfilter([np.sqrt(1 - b[3] ** 2)], [1 - b[3]], w))
    train_dataset = torch.cat([q0, q1, q2, q3], dim=1)

    w = np.random.randn(len_val, 1) + 0.03
    w = w / np.max(w)
    q0 = torch.FloatTensor(lfilter([np.sqrt(1 - b[0] ** 2)], [1 - b[0]], w))
    q1 = torch.FloatTensor(lfilter([np.sqrt(1 - b[1] ** 2)], [1 - b[1]], w))
    q2 = torch.FloatTensor(lfilter([np.sqrt(1 - b[2] ** 2)], [1 - b[2]], w))
    q3 = torch.FloatTensor(lfilter([np.sqrt(1 - b[3] ** 2)], [1 - b[3]], w))
    val_dataset = torch.cat([q0, q1, q2, q3], dim=1)

    w = np.random.randn(len_test, 1) + 0.03
    w = w / np.max(w)
    q0 = torch.FloatTensor(lfilter([np.sqrt(1 - b[0] ** 2)], [1 - b[0]], w))
    q1 = torch.FloatTensor(lfilter([np.sqrt(1 - b[1] ** 2)], [1 - b[1]], w))
    q2 = torch.FloatTensor(lfilter([np.sqrt(1 - b[2] ** 2)], [1 - b[2]], w))
    q3 = torch.FloatTensor(lfilter([np.sqrt(1 - b[3] ** 2)], [1 - b[3]], w))
    test_dataset = torch.cat([q0, q1, q2, q3], dim=1)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset)
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset)
    )

    return train_loader, val_loader, test_loader