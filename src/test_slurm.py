
import os

import time
tstamp = time.localtime().tm_min
# tstamp = '240407_0705h'

file = f'{os.getcwd()}/src/test_slurm_{tstamp}.txt'
if os.path.exists(file):
    os.remove(file)

with open(file, 'w') as f:
    f.write('test file\n')

