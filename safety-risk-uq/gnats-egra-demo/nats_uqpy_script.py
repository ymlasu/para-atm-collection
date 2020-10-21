"""Script to handle running NATS for UQpy

Refer to https://uqpyproject.readthedocs.io/en/latest/runmodel_doc.html
"""

import os
import sys
import subprocess
import fire

def model(sample_index):

    input_file = os.path.join('InputFiles','nats_input_{}.txt'.format(sample_index))
    output_file = 'nats_output_{}.txt'.format(sample_index)

    # sys.executable refers to the current Python executable
    subprocess.run([sys.executable, 'nats_driver.py', input_file, output_file])
    
if __name__ == '__main__':
   fire.Fire(model)
