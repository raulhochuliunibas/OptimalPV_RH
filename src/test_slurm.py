
import os
import time

miniute_of_hour = time.localtime().tm_min

file = f'{os.getcwd()}/code/test_slurm_{miniute_of_hour}.txt'
if os.path.exists(file):
    os.remove(file)

with open(file, 'w') as f:
    f.write('test file\n')

