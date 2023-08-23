import os as os
from datetime import datetime

def chapter_to_logfile(str):
    """
    Function to write a chapter to the logfile
    """
    check = f'\n\n****************************************\n {str} \n start at:{datetime.now()} \n****************************************\n\n'
    print(check)
    with open(f'log_file.txt', 'a') as log_file:
        log_file.write(f'{check}\n')



def checkpoint_to_logfile(str):
    """
    Function to write a checkpoint to the logfile
    """
    check = f'* {str}: {datetime.now()}'
    print(check)
    with open(f'log_file.txt', 'a') as log_file:
        log_file.write(f"{check}\n")


# ----------------------------------------------------------------------
# book mark ------------------------------------------------------------
# ----------------------------------------------------------------------