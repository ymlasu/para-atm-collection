"""Script that implements simulation output retrieval for UQpy

Refer to https://uqpyproject.readthedocs.io/en/latest/runmodel_doc.html
"""

import os

def get_output(sample_index):

    output_file = 'nats_output_{}.txt'.format(sample_index)

    # print('get_output:', os.path.abspath(output_file))
    with open(output_file,'r') as f:
        val = float(f.readline().strip())

    return val
