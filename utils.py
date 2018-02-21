#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import sys

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def mkdir_unique(timestamp=get_timestamp(), prefix=''):
    mydir = os.path.join(
        os.getcwd(), 
        prefix + '_' + timestamp)

    os.makedirs(mydir)

    return mydir

# Print iterations progress
def print_progress_bar(iteration, total, 
	prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

