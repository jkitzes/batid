#!/usr/bin/python

'''
Minimal script
'''

import os
import datetime, time
from param_rw import read_params

def main():

    #1/0
    time.sleep(5)

    # Read query_dict from parameters file in current directory
    query_dict = read_params(os.getcwd())

    # Write some output
    with open('answer.txt', 'w') as f:
        f.write('The answer at ' + str(datetime.datetime.now()))
        f.write(str(query_dict))

if __name__ == '__main__':
    main()
