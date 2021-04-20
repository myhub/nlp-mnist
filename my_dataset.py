import torch
import torch.nn as nn
import time
import os
import argparse
# import random
from random import SystemRandom
random = SystemRandom()
from torch.utils.data import DataLoader, Dataset
import sys
from io import StringIO

class MyDataset(Dataset): 
    def __init__(self):
        self.a_z = []
        self.digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for i in range(26):
            self.a_z += chr(i + ord('a'))

        self.BLANK = chr(2)
        self.alphabet = [self.BLANK] + self.digits + self.a_z + [' ', ';', '=', '(', ')']

        self.AIN_SIZE = 64
        # self.AIN_SIZE = 60
        # self.AOUT_SIZE = 32

        self.char2idx = {}
        for i, c in enumerate(self.alphabet):
            self.char2idx[c] = i


    def __len__(self):
        return 1024
    
    def __getitem__(self, index):
        _ = index
        sel = random.sample(self.a_z, k=random.randint(2, 8))

        ain = ""
        for k in sel:
            ain += k
            ain += '=' + str(random.randint(0, 9))
            ain += ";"

        stdout_new = StringIO()
        stdout_save = sys.stdout

        loc = {}
        name = random.choice(sel)
        ain += "print(" + name + ') '
        sys.stdout = stdout_new
        exec(ain, None, loc)
        sys.stdout = stdout_save

        aout = stdout_new.getvalue().strip()

        ain = self.BLANK * (self.AIN_SIZE - len(ain)) + ain
        # aout = aout + self.BLANK * (self.AOUT_SIZE - len(aout))

        aout = aout[0]
        
        ain = [self.char2idx[_] for _ in ain]
        aout = [self.char2idx[_] for _ in aout]
        return torch.LongTensor(ain), torch.LongTensor(aout)


my_dataset = MyDataset()

if __name__ == '__main__':
    from torch import nn 

    ds = MyDataset()

    for i in range(15):
        ain, aout = ds[i]
        print(' | ', end="")
        for idx in ain:
            c = ds.alphabet[idx]
            if c == ds.BLANK: continue
            print(ds.alphabet[idx], end='')
        print(' | ', end="")
        for idx in aout:
            c = ds.alphabet[idx]
            if c == ds.BLANK: continue
            print(ds.alphabet[idx], end='')
        print(' | ')

