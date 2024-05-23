#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/gdrive')
# %cd '/content/gdrive/My Drive'




from utils.tools import classes
from utils.agent import *
from utils.dataset import read_voc_dataset
import tqdm.notebook as tq
import pickle

use_depth = True
eval_baseline = True


dst_dir = 'results'

if use_depth:
    dst_dir = 'results_depth'
    classes_baseline = classes.copy() + ['closest']
    classes = ['closest']

# model_name='resnet50_3step'
model_name = 'vgg16_3step'

_, val_loader2012 = read_voc_dataset(path="./data/PascalVOC2012", year='2012', download=True)

if not use_depth:
    datasets_per_class = sort_class_extract([val_loader2012])
else:
    datasets_per_class = sort_class_extract_depth([val_loader2012])


torch.cuda.empty_cache()
results = {}
for i in classes:
    results[i] = []

if  eval_baseline and use_depth:
    for i in tq.tqdm(range(len(classes_baseline))):
        curr_class = classes_baseline[i]
        print("Baseline Class: " + str(curr_class) + "...")
        agent = Agent_3alpha(curr_class, load=True, model_name=model_name)
        res = agent.evaluate(datasets_per_class['closest'], use_depth=use_depth)
        results[curr_class] = res
        print(results)

else:
    for i in tq.tqdm(range(len(classes))):
        curr_class = classes[i]
        print("Class: " + str(curr_class) + "...")
        agent = Agent_3alpha(curr_class, load=True, model_name=model_name)
        res = agent.evaluate(datasets_per_class[curr_class], use_depth=use_depth)
        results[curr_class] = res
        print(results)


file_name = f'{dst_dir}/classes_results_' + model_name + '.pickle'
os.makedirs(f'{dst_dir}', exist_ok=True)
with open(file_name, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## Visualize Random Class


# curr_class = random.choice(classes)
# indices = np.random.choice(list(datasets_per_class[curr_class].keys()), size=5, replace=False)
# agent = Agent_3alpha(curr_class, load=True, model_name=model_name)

# print("Class: " + curr_class)
# for index in indices:
#     image, gt_boxes = extract(index, datasets_per_class[curr_class])
#     agent.predict_image(image, plot=True)




