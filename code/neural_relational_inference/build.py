import os
from pathlib import Path
import pandas as pd
import datetime
import uuid
import pickle

from mgl.neural_relational_inference.build import *

path_to_datacache = Path('../scratch')
path_to_data = Path('../data')
path_to_save = Path('../results')

# find where the data are stored
for root, dirs, files in os.walk(path_to_data):
    if len([f for f in files if 'model_params_' in f]): 
        path_to_param_data = Path(root)
        break

# look for session specific data related to model to be trained
model_param_files = [f for f in os.listdir(path_to_param_data) if 'model_params_' in f]
#assert len(model_param_files)==1 # if not 1, there was an error in parallelizing

# import hyperparameter
f = path_to_param_data / model_param_files[0]
with open(f, "rb") as input_file:
    model_params = pickle.load(input_file)


ophys_session_id = int(model_params["ophys_session_id"])

# used to grab session data from different capsule
path_to_session_data = Path(model_params["path_to_session_data"])
if not path_to_session_data.is_dir():
    # find where the data are stored
    for root, dirs, files in os.walk(path_to_data):
        if len([f for f in files if 'tensor' in f]): break 
        temp_path_to_data = Path(root)
    
    session_data_dirs = [f for f in os.listdir(temp_path_to_data) if str(ophys_session_id) in f]
    path_to_session_data = temp_path_to_data / session_data_dirs[0]

filename = [f for f in os.listdir(path_to_session_data) if '.npy' in f][0]
f = path_to_session_data / filename
session_tensor = np.load(f)

# update hyperparameter file for model
now = datetime.datetime.now()
timestamp = now.isoformat()
nri_experiment_id = str(uuid.uuid4())

param_label = 'param_set_%s'%model_params["param_set_id"]
experiment_label = 'exp_%s'%nri_experiment_id
path_to_model_exp_save = path_to_save / param_label / experiment_label
path_to_model_exp_save.mkdir(parents=True, exist_ok=True)


model_params.update({'nri_experiment_id': nri_experiment_id,
                    'experiment_date_time' : timestamp,
                    'path_to_save' : str(path_to_model_exp_save) # needed for downstream capsule to fine the models
                    })

# create arg parser
parser = generate_hyperparam_parser_from_dict(model_params)
args = parser.parse_args("")

# Get train-validation-test split
# FUTURE: Need a version of this for connectivity as well
print("Grabing training, validation, and test batches...")
train_data_loader, valid_data_loader, test_data_loader = reformat_data_for_model_training(session_tensor,
                                                                                          batch_size=args.batch_size,
                                                                                          verbose=True)

# train NRI model
print("Training NRI model and saving...")
train_performance_df, best_model_files = initialize_and_train_NRI_model(args,train_data_loader,
                                                                        valid_data_loader,verbose=True)


f = path_to_model_exp_save / 'train_performance.pkl'
train_performance_df.to_pickle(f)

# test NRI model
print("Testing NRI model and saving...")
test_performance_df = initialize_and_test_NRI_model(args,test_data_loader,
                                                    best_model_files,verbose=True)

f = path_to_model_exp_save / 'test_performance.pkl'
test_performance_df.to_pickle(f)
