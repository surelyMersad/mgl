import os
import sys
from pathlib import Path

import uuid
import argparse
import pickle
import datetime
from pprint import pprint

import pandas as pd
import numpy as np

from mgl.neural_relational_inference.build import *



###########################
#### Global parameters ####
###########################

# I/O 
path_to_save = Path('../results')

# find where the data are stored
for root, dirs, files in os.walk('../data'):
    if len([f for f in files if '.npy' in f]): break 
    path_to_data = Path(root)

session_data_dirs = [f for f in os.listdir(path_to_data) if 's' in f]
#assert len(session_data_dirs)==1 
path_to_session_data = path_to_data / session_data_dirs[0]


# edge parameters
connection_prior_model = 'uniform_sparsity_prior_dist' # FUTURE: define different classes of connectivity priors 
                                                       # (e.g., uniform sparsity prior vs non-uniform sparsity prior distributions)
    
connection_sparsity = 0.84
num_edge_types_list = [2,3,4,5,6] # use to generate multiple model sweeps - fine tune interaction graph


# training parameters
num_epochs = 500 # FUTURE: use to generate multiple model sweeps -- fine tune compute time/performance 
batch_size = 128 # FUTURE: use to generate multiple model sweeps -- fine tune compute time/performance

# model architecture
train_encoder = True
encoder_model_class = 'mlp' # mlp or cnn
decoder_model_class = 'mlp' # mlp or rnn

# whether or not to include empirical/ground truth connectivity data
connectivity = False # if train_encoder=False and connectivity=True, then only the decoder should be used
                     # train_encoder and connectivity cannot both be False, but both can be True if testing edge accuracy across training

# CUDA 
parallelize = False





##############################
#### Parameter generation ####
####   and versioning     ####
##############################

f = path_to_session_data / 'metadata.pkl'
metadata_df = pd.read_pickle(f)

mouse_id = metadata_df.mouse_id.iloc[0]
ophys_session_id = metadata_df.ophys_session_id.iloc[0]

# find and import the numpy file with trial data
filename = [f for f in os.listdir(path_to_session_data) if '.npy' in f][0]
session_tag, ophys_signal, _, stim_window = filename.split('_')

# FUTURE: Will generalize for multiple session tensors for different ophys signals (ex. DFF and events). But for now, assumes 1 ophys signal type.
f = path_to_session_data / filename
session_tensor = np.load(f)

# node parameters
num_features = 1
ophys_signals = [ophys_signal] # dff or events or filtered_events

if len(ophys_signals)!=num_features:
    raise Exception("Number of features much match number of ophys signals")

# data parameters
num_cells, windowed_timesteps, num_windows = np.shape(session_tensor)

total_timesteps = windowed_timesteps
prediction_steps = int(0.4*total_timesteps)


for i, num_edge_types in enumerate(num_edge_types_list):

    param_set_id = i+1

    # base model experiment template
    model_params = {'param_set_id' : param_set_id,
                        'num_edge_types' : num_edge_types, # default: 2 => connected or not connected
                        'connectivity' : connectivity, # true or false
                        'num_ophys_signals' : num_features, # number of node features (ex. dff + events = 2)
                        'ophys_signals' : ophys_signals,
                        'include_encoder' : train_encoder,
                        'encoder' : encoder_model_class if train_encoder else None,
                        'decoder' : decoder_model_class,
                        'is_dynamic' : True if decoder_model_class=='rnn' else False,
                        'timesteps' : total_timesteps,
                        'prediction_steps' : prediction_steps,
                        'prior_model' : connection_prior_model,
                        'prior' : get_uniform_sparsity_prior_distribution(connection_sparsity,num_edge_types),
                        'num_epochs' : num_epochs,
                        'batch_size' : batch_size,
                        'CUDA' : parallelize,
                        'path_to_session_data' : str(path_to_session_data) # needed by downstream capsules to find original data
                        }

    # update for specific sessions
    model_params.update({'mouse_id' : mouse_id,
                        'ophys_session_id' : ophys_session_id,
                        'num_nodes' : num_cells,
                        })

    # save model parameters
    filename = 'model_params_'+'s'+str(model_params['ophys_session_id'])+'_'+str(model_params['param_set_id'])+'.pkl'
    f = path_to_save / filename

    with open(f, "wb") as outfile: 
        pickle.dump(model_params, outfile)


            
