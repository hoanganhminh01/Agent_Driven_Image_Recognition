import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from config import *
import pandas as pd
import random
import tqdm.notebook as tq
from PIL import Image

# Animals
classes_animals = ['cat', 'cow', 'dog', 'bird', 'horse', 'sheep', 'person'] 
# classes_animals = ['cat', 'cow', 'dog', 'bird', 'horse', 'sheep'] 
# classes_animals = ['person'] 

# Objects
classes_objects = ['bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
# Vehicles
classes_vehicles = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train']
classes = classes_animals
depth_classes = classes_animals

def sort_class_extract(datasets):
    """
    Change a whole dataset to seperate datasets corresponding to classes variable
    """
    datasets_per_class = {}
    for j in classes:
        datasets_per_class[j] = {}

    for dataset in datasets:
        for i in tq.tqdm(dataset):
            img, target = i

            obj = target['annotation']['object']
            if isinstance(obj, list):
                curr_class = target['annotation']['object'][0]["name"]
            else:
                curr_class = target['annotation']['object']["name"]
            filename = target['annotation']['filename']

            org = {}
            for j in classes:
                org[j] = []
                org[j].append(img)

            if isinstance(obj, list):
                for j in range(len(obj)):
                    curr_class = obj[j]["name"]
                    if curr_class in classes:
                        org[curr_class].append([obj[j]["bndbox"], target['annotation']['size']])
                        # if len(org[curr_class]) > 2:
                        #     print(org[curr_class])
                        #     print("More than one object in the same image")
                        #     exit()
            else:
                if curr_class in classes:
                    org[curr_class].append([obj["bndbox"], target['annotation']['size']])

           

            for j in classes:
                if len(org[j]) > 1:
                    try:
                        datasets_per_class[j][filename].append(org[j])
                    except KeyError:
                        datasets_per_class[j][filename] = []
                        datasets_per_class[j][filename].append(org[j])   

        
    # print((datasets_per_class.keys()))
    # for i in classes:
    #     print(len(datasets_per_class[i].keys()))
    # exit()

    return datasets_per_class




def sort_class_extract_depth(datasets):
    """
    Change a whole dataset to seperate datasets corresponding to classes variable
    """
    datasets_per_class = {}
    datasets_per_class['closest'] = {}
    ind = 0

    for dataset in datasets:
        for i in tq.tqdm(dataset):
            img, target = i

            obj = target['annotation']['object']
            if isinstance(obj, list):
                curr_class = target['annotation']['object'][0]["name"]
            else:
                curr_class = target['annotation']['object']["name"]
            filename = target['annotation']['filename']

            # HACK reading the depth
            depth_dir = './data/PascalVOC2012/VOCdevkit/VOC2012/depth_maps'
            depth_path = f'{depth_dir}/{filename[:-4]}.png'
            depth_img = Image.open(depth_path).convert('L').resize((224, 224), Image.NEAREST)
            depth_img = np.array(depth_img)

            # print(depth_img.shape)
            # print(img.shape)
            # exit()
            # print(depth_classes)

            org = {}
            for j in depth_classes:
                org[j] = []
                org[j].append(img)

            if isinstance(obj, list):
                # get depth of each bbox 
                depths = []
                ratio = 0.8
                for j in range(len(obj)):
                    bbox = obj[j]["bndbox"]
                    xmin = int(bbox['xmin'])
                    xmax = int(bbox['xmax'])
                    ymin = int(bbox['ymin'])
                    ymax = int(bbox['ymax'])

                    ori_width = target['annotation']['size']['width']
                    ori_height = target['annotation']['size']['height']

                    ori_width = float(ori_width)
                    ori_height = float(ori_height)

                    xmin = xmin /  ori_width * 224
                    xmax = xmax /  ori_width * 224

                    ymin = ymin /  ori_height * 224
                    ymax = ymax /  ori_height * 224

                    # shrink the size of  bbox according to the ratio
                    width = xmax - xmin
                    height = ymax - ymin

                    xmin = int(xmin + width * (1-ratio)/2)
                    xmax = int(xmax - width * (1-ratio)/2)

                    ymin = int(ymin + height * (1-ratio)/2)
                    ymax = int(ymax - height * (1-ratio)/2)

        

                    depth = depth_img[ymin:ymax, xmin:xmax].mean()
                    depths.append(depth)

                # print(depths)


                # only take classes with more than 2 objects
                if len(obj) > 1:

                    # get max depth ind and save it to org
                    max_depth_ind = np.argmax(depths)
                    curr_class = obj[max_depth_ind]["name"]

                    if curr_class in depth_classes:
                        
                        org[curr_class].append([obj[max_depth_ind]["bndbox"], target['annotation']['size'], depths[max_depth_ind]])

                        # visualization
                        visualization = False
                        if visualization:
                                
                            ind += 1
                            bbox = obj[max_depth_ind]["bndbox"]
                            xmin = int(bbox['xmin'])
                            xmax = int(bbox['xmax'])
                            ymin = int(bbox['ymin'])
                            ymax = int(bbox['ymax'])


                            ori_width = float(ori_width)
                            ori_height = float(ori_height)

                            xmin = xmin /  ori_width * 224
                            xmax = xmax /  ori_width * 224

                            ymin = ymin /  ori_height * 224
                            ymax = ymax /  ori_height * 224

                            show_new_bdbox(img,[xmin, xmax, ymin, ymax], count=ind)


                

                
                # for j in range(len(obj)):
                #     curr_class = obj[j]["name"]
                #     if curr_class in classes:
                #         org[curr_class].append([obj[j]["bndbox"], target['annotation']['size']])
                        
            else:
                if curr_class in depth_classes:
                    org[curr_class].append([obj["bndbox"], target['annotation']['size']])
            
           
            # save everything into the class 'closest' to the camera
            for j in depth_classes:
                if len(org[j]) > 1:
                    try:
                        datasets_per_class['closest'][filename].append(org[j])
                        
                    except KeyError:
                        datasets_per_class['closest'][filename] = []
                        datasets_per_class['closest'][filename].append(org[j])   


    
    print((datasets_per_class.keys()))
    print(len(datasets_per_class['closest'].keys()))
    # exit()

    return datasets_per_class



def show_new_bdbox(image, labels, color='r', count=0):
    """
    Imshow an image and corresponding bounding box
    """
    xmin, xmax, ymin, ymax = labels[0],labels[1],labels[2],labels[3]
    fig,ax = plt.subplots(1)
    ax.imshow(image.transpose(0, 2).transpose(0, 1))
    import os
    width = xmax-xmin
    height = ymax-ymin
    rect = patches.Rectangle((xmin,ymin),width,height,linewidth=3,edgecolor=color,facecolor='none')
    ax.add_patch(rect)
    ax.set_title("Iteration "+str(count))
    os.makedirs('./temp', exist_ok=True)
    plt.savefig('./temp/'+str(count)+'.jpg', dpi=100)


def extract(index, loader):
    """
    Extract image and ground truths from a data loader
    ----------
    Argument:
    index              - the index of the img, should be a string of its filename, e.g '00001.jpg'
    loader             - an instance of data loader, 
                         should be a large dict, each key is image filename, value is its information
    ----------
    Return:
    img                - image value, should be (3,224,224)
    ground_truth_boxes - a list of ground truth boxes in this image, 
                         length of this list should equal to how many objects there are in this image
    """
    extracted = loader[index]
    img = extracted[0][0]
    ground_truth_boxes =[]
    for ex in extracted[0][1:]:
        bndbox = ex[0]
        size = ex[1]
        xmin = ( float(bndbox['xmin']) /  float(size['width']) ) * 224
        xmax = ( float(bndbox['xmax']) /  float(size['width']) ) * 224

        ymin = ( float(bndbox['ymin']) /  float(size['height']) ) * 224
        ymax = ( float(bndbox['ymax']) /  float(size['height']) ) * 224

        ground_truth_boxes.append([xmin, xmax, ymin, ymax])
    return img, ground_truth_boxes


def voc_ap(rec, prec, voc2012=True):
    if voc2012:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
            print(p)

    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def intersection_over_union(box1, box2):
        """
        Compute IoU value over two bounding boxes
        Each box is represented by four elements vector: (left, right, top, bottom)
        Origin point of image system is on the top left
        """
        box1_left, box1_right, box1_top, box1_bottom = box1
        box2_left, box2_right, box2_top, box2_bottom = box2
        
        inter_top = max(box1_top, box2_top)
        inter_left = max(box1_left, box2_left)
        inter_bottom = min(box1_bottom, box2_bottom)
        inter_right = min(box1_right, box2_right)
        inter_area = max(((inter_right - inter_left) * (inter_bottom - inter_top)), 0)
        
        box1_area = (box1_right - box1_left) * (box1_bottom - box1_top)
        box2_area = (box2_right - box2_left) * (box2_bottom - box2_top)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

def prec_rec_compute(bounding_boxes, gt_boxes, ovthresh):

    
    nd = 0
    for each in gt_boxes:
        nd += len(each)
    npos = nd
    
    ious = np.zeros(nd)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    d = 0

    for index in range(len(bounding_boxes)):
        for gtbox in gt_boxes[index]:
            best_iou = 0
            for bdbox in bounding_boxes[index]:
                iou = intersection_over_union(gtbox,bdbox)
                if iou > best_iou:
                    best_iou = iou
                    
            if best_iou > ovthresh:
                tp[d] = 1.0
            else:            
                fp[d] = 1.0
                
            ious[d] = best_iou
            
            d += 1
            
    sort_idx = np.argsort(-ious)
    fp = fp[sort_idx]
    tp = tp[sort_idx]
    ious = ious[sort_idx]
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return prec, rec


def compute_ap_and_recall(all_bdbox, all_gt, ovthresh):
    prec, rec = prec_rec_compute(all_bdbox, all_gt, ovthresh)
    ap = voc_ap(rec, prec, True)
    return ap, rec[-1]


def eval_stats_at_threshold(all_bdbox, all_gt, thresholds=[0.4, 0.5, 0.6]):
    stats = {}
    for ovthresh in thresholds:
        ap, recall = compute_ap_and_recall(all_bdbox, all_gt, ovthresh)
        stats[ovthresh] = {'ap': ap, 'recall': recall}
    stats_df = pd.DataFrame.from_records(stats)*100
    return stats_df


class ReplayMemory(object):
    """
    A replay memory object to do expereience replay.
    Each sample is 
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
