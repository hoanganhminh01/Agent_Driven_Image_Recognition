from utils.agent import *
from utils.dataset import read_voc_dataset
from IPython.display import clear_output
import matplotlib.pyplot as plt
import tqdm.notebook as tq
import numpy as np

train_loader2007, val_loader2007 = read_voc_dataset(
    path="./data/PascalVOC2012", year="2012", download=False
)
datasets_per_class = sort_class_extract(
    [
        train_loader2007,
    ]
)
cat_dataset = datasets_per_class["cat"]
print(len(cat_dataset))
t = 1

# plotting

# plt.figure(figsize=[15, 6])
# indexes = []
# for i in list(cat_dataset.keys())[:10]:
#     indexes.append(i)
#     image = cat_dataset[i][0][0]
#     plt.subplot(2, 5, t)
#     plt.imshow(image.permute([1, 2, 0]))

#     for j in range(1, len(cat_dataset[i][0])):
#         gt = cat_dataset[i][0][j][0]
#         image_size = cat_dataset[i][0][j][1]

#         origin_width, origin_height = int(image_size["width"]), int(
#             image_size["height"]
#         )
#         real_width, real_height = 224, 224
#         width_ratio, height_ratio = (
#             real_width / origin_width,
#             real_height / origin_height,
#         )

#         bdbox = np.array([gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"]]).astype(
#             "float"
#         )
#         bdbox = (bdbox * [width_ratio, height_ratio, width_ratio, height_ratio]).astype(
#             "int"
#         )

#         plt.gca().add_patch(
#             plt.Rectangle(
#                 (bdbox[0], bdbox[1]),
#                 bdbox[2] - bdbox[0],
#                 bdbox[3] - bdbox[1],
#                 fill=False,
#                 edgecolor="r",
#                 linewidth=3,
#             )
#         )
#     t += 1


def train(model_type, eps, BATCH_SIZE=64):
    agents = []
    for i in tq.tqdm(range(len(classes))):
        classe = classes[i]
        print("Classe " + str(classe) + "...")

        if model_type == "org":
            agent = Agent(
                classe,
                alpha=0.2,
                num_episodes=eps,
                load=False,
                model_name="vgg16_org",
                n_actions=9,
                BATCH_SIZE=BATCH_SIZE,
            )
        elif model_type == "3_step":
            agent = Agent_3alpha(
                classe,
                alpha=0.2,
                num_episodes=eps,
                nu=10.0,
                threshold=0.65,
                load=False,
                model_name="vgg16_3step",
                n_actions=25,
                BATCH_SIZE=BATCH_SIZE,
            )

        agent.train(datasets_per_class[classe])
        agents.append(agent)
        del agent
        torch.cuda.empty_cache()
    return agents


from utils.tools import classes


def test(model, model_name, data_set):

    # Test on whole dataset to get result metrics
    torch.cuda.empty_cache()
    results = {}
    for i in classes:
        results[i] = []

    for i in tq.tqdm(range(len(classes))):
        classe = classes[i]
        print("Class " + str(classe) + "...")
        agent = model
        res = agent.evaluate(cat_dataset)
        results[classe] = res


def visualize(img_num, agent):
    toy_val = {}
    for i, (k, v) in enumerate(datasets_per_class["cat"].items()):
        if i == img_num:
            toy_val[k] = v
            break

    for key, value in toy_val.items():
        image, gt_boxes = extract(key, toy_val)
        bbox = agent.predict_multiple_objects(image, plot=True)
        break


def main():
    agent_3step = train("3_step", 15)[0]
    agent_org = train("org", 15)[0]
    print(agent_3step)
    print(agent_org)

    print(test(agent_3step, "vgg16_3step", cat_dataset))
    print(test(agent_org, "vgg16_org", cat_dataset))

    datasets_per_class_val = sort_class_extract([val_loader2007])

    visualize(8, agent_3step)
    torch.cuda.empty_cache()
    model_name = "vgg16_org"
    agent = Agent(classe, load=True, model_name=model_name)
    res = agent.evaluate(datasets_per_class_val["cat"])
    torch.cuda.empty_cache()
    model_name = "vgg16_3step"
    # agent = Agent(classe, load=True, model_name=model_name)
    res = agent_3step.evaluate(datasets_per_class_val["cat"])
    print(ground_truth_boxes)


if __name__ == "__main__":
    main()
