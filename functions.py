import os as os
from datetime import datetime

def chapter_to_logfile(str, log_file_name):
    """
    Function to write a chapter to the logfile
    """
    check = f'\n\n****************************************\n {str} \n start at:{datetime.now()} \n****************************************\n\n'
    print(check)
    with open(f'{log_file_name}', 'a') as log_file:
        log_file.write(f'{check}\n')


time_last_call = None

def checkpoint_to_logfile(str, log_file_name, n_tabs = 0, timer_func=None):
    """
    Function to write a checkpoint to the logfile
    """
    global time_last_call
    
    time_now = datetime.now()
    if time_last_call:
        runtime = time_now - time_last_call
        minutes, seconds = divmod(runtime.seconds, 60)
        runtime_str = f"{minutes} min {seconds} sec"
    else:
        runtime_str = 'N/A'
    
    n_tabs_str = '\t' * n_tabs
    check = f'* {str}{n_tabs_str}runtime: {runtime_str};   (stamp: {datetime.now()})'
    print(check)

    with open(f'{log_file_name}', 'a') as log_file:
        log_file.write(f"{check}\n")
    
    time_last_call = time_now


# ----------------------------------------------------------------------
# book mark ------------------------------------------------------------
# 
# ----------------------------------------------------------------------
    
def test_functions(text):
    print(f'return: {text}')