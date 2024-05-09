#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/gdrive')
# %cd '/content/gdrive/My Drive'


# In[2]:


from utils.tools import classes
from utils.agent import *
from utils.dataset import read_voc_dataset
import tqdm.notebook as tq
import pickle


# In[3]:



# In[1]:

model_name='resnet50_3step'

_, val_loader2012 = read_voc_dataset(path="./data/PascalVOC2012", year='2012', download=True)


# In[2]:


datasets_per_class = sort_class_extract([val_loader2012])


# In[4]:


torch.cuda.empty_cache()
results = {}
for i in classes:
    results[i] = []
for i in tq.tqdm(range(len(classes))):
    curr_class = classes[i]
    print("Class: " + str(curr_class) + "...")
    agent = Agent_3alpha(curr_class, load=True, model_name=model_name)
    res = agent.evaluate(datasets_per_class[curr_class])
    results[curr_class] = res


# In[ ]:


file_name = 'results/classes_results_' + model_name + '.pickle'
os.makedirs('results', exist_ok=True)
with open(file_name, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## Visualize Random Class

# In[6]:


# curr_class = random.choice(classes)
# indices = np.random.choice(list(datasets_per_class[curr_class].keys()), size=5, replace=False)
# agent = Agent_3alpha(curr_class, load=True, model_name=model_name)

# print("Class: " + curr_class)
# for index in indices:
#     image, gt_boxes = extract(index, datasets_per_class[curr_class])
#     agent.predict_image(image, plot=True)


# In[ ]:




