from utils.tools import classes
from utils.agent import *
from utils.dataset import read_voc_dataset
from IPython.display import clear_output
import tqdm.notebook as tq
import pickle
import fire
# vgg19_3step

def main(model_name, BATCH_SIZE=128):

    # hack 
    if '_3step' not in model_name:
        model_name = model_name + '_3step'

    train_loader2012, val_loader2012 = read_voc_dataset(
        path="./data/PascalVOC2012", year="2012", download=False
    )

    datasets_per_class_train = sort_class_extract([train_loader2012])
    datasets_per_class_test = sort_class_extract([val_loader2012])

    for i in tq.tqdm(range(len(classes))):
        curr_class = classes[i]
        print("Class: " + str(curr_class) + "...")
        # agent = Agent(classe, alpha=0.2, num_episodes=15, load=False, model_name='vgg16')
        agent = Agent_3alpha(
            curr_class,
            alpha=0.2,
            num_episodes=15,
            load=False,
            model_name=model_name,
            BATCH_SIZE=BATCH_SIZE,
        )
        agent.train(datasets_per_class_train[curr_class])
        del agent
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    results = {}
    for i in classes:
        results[i] = []
    # model_name = "vgg19_3step"
    for i in tq.tqdm(range(len(classes))):
        curr_class = classes[i]
        print("Class: " + str(curr_class) + "...")
        agent = Agent_3alpha(curr_class, load=True, model_name=model_name, BATCH_SIZE=BATCH_SIZE)
        res = agent.evaluate(datasets_per_class_test[curr_class])
        results[curr_class] = res

    file_name = "classes_results_" + model_name + ".pickle"
    with open(file_name, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    curr_class = random.choice(classes)
    indices = np.random.choice(
        list(datasets_per_class_test[classe].keys()), size=5, replace=False
    )
    agent = Agent_3alpha(curr_class, load=True, model_name=model_name)

    print("Class: " + curr_class)
    for index in indices:
        image, gt_boxes = extract(index, datasets_per_class_test[curr_class])
        agent.predict_image(image, plot=True)


if __name__ == "__main__":
    fire.Fire(main)
