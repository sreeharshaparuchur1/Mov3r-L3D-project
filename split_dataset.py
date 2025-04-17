import json
import numpy as np

import os
dataset_path = '/data/kmirakho/l3dProject/scannetv2/'

#create a json file to slipt dataset of 100 trajectories into train and val of 80 and 20 arranged randomly
list_of_trajectories = os.listdir(dataset_path)
np.random.shuffle(list_of_trajectories)
train_trajectories = list_of_trajectories[:int(len(list_of_trajectories)*0.8)]
val_trajectories = list_of_trajectories[int(len(list_of_trajectories)*0.8):]

print("hello")
#save the train and val trajectories in a json file
with open('train_trajectories.json', 'w') as f:
    json.dump(train_trajectories, f)
with open('val_trajectories.json', 'w') as f:
    json.dump(val_trajectories, f)