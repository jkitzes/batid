#!/usr/bin/python

'''
Utility functions for bat analysis.
'''

import numpy as np
import os
import ConfigParser

def write_params(query_dict, output_dir):
    '''Save dict as config.cfg parameter file.'''

    # Set up config file
    config = ConfigParser.RawConfigParser()
    config.add_section('Section')
    for key, val in query_dict.items():
        config.set('Section', key, val)
   
    # Write config file to output_path
    with open(os.path.join(output_dir, 'config.cfg'), 'wb') as configfile:
        config.write(configfile)


def read_params(input_dir):
    '''Read parameter file from config.cfg as dict.'''
    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(input_dir, 'config.cfg'))
    return dict(config.items('Section'))


def sum_group(ids, table=None, others=None):
    '''
    Sums all rows of table with the same group, returning a set of unique ids 
    and associated summed rows from table.

    http://stackoverflow.com/questions/4373631/sum-array-by-number-in-numpy

    Others is list of other vectors to be grouped like ids.
    '''
   
    ids = np.array(ids)  # "Off by one" comparison only works on arrays

    last_id = np.ones(len(ids), 'bool')
    last_id[:-1] = (ids[1:] != ids[:-1])
    ids_grouped = ids[last_id]

    if table != None:
        table_cum = table.cumsum(axis = 0)
        table_grouped = table_cum[last_id]
        table_grouped[1:] = (table_grouped[1:] - table_grouped[:-1])
    else:
        table_grouped = None

    if others != None:
        others_grouped = []
        for i, other in enumerate(others):
            other_array = np.array(other)
            others_grouped.append(other_array[last_id])
    else:
        others_grouped = None

    return ids_grouped, table_grouped, others_grouped
