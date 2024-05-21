# Some basic setup:
# Setup detectron2 logger
import os
import sys
# include the path to the root directory of the project

import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog





from utils.tools import classes
from utils.agent_baseline import *
from utils.dataset import read_voc_dataset
import pickle

model_name = 'PascalVOC-Detection/faster_rcnn_R_50_C4.yaml'

_, val_loader2012 = read_voc_dataset(path="./data/PascalVOC2012", year='2012', download=True)


# In[2]:


datasets_per_class = sort_class_extract([val_loader2012])

# print(datasets_per_class)




torch.cuda.empty_cache()
results = {}
for i in classes:
    results[i] = []
for i in tq.tqdm(range(len(classes))):
    curr_class = classes[i]
    print("Class: " + str(curr_class) + "...")
    agent = Agent_3alpha(curr_class, load=True, model_name=model_name)
    res = agent.evaluate(datasets_per_class[curr_class], curr_class)
    results[curr_class] = res
    print(results)

    # exit()



file_name = 'results_baseline/classes_results_' + model_name + '.pickle'
os.makedirs(os.path.dirname(file_name), exist_ok=True)
with open(file_name, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)




