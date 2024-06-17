import os
import glob
import random
import sys
import json
import constants,utils,xpreprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import concurrent.futures



dataset_root = os.path.join(".", "Datasets")
dn = constants.dataset_name
af = constants.aug_factor
batch_generate_size = 1
batch_pseudo_generate_size = 2
step=4
save_path = "./Datasets/COMPRESS-DATA/"
size=8


def process_data():

        read_dir = os.path.join(dataset_root, f"TRAIN-DATA")
        read_img_paths = glob.glob(os.path.join(read_dir, "*.png"), recursive=False)
        # print(read_img_paths)

        groups = [read_img_paths[i:i + 81] for i in range(0, len(read_img_paths), 81)]
        random.shuffle(groups)
        shuffled_img_paths = [img for group in groups for img in group] 

        sam_labels_path = os.path.join('./Datasets/', 'labels', 'DUT-LF_SAM_labels.json')
        with open(sam_labels_path, 'r') as file:
            labels = json.load(file)

        group_sum = len(labels)
        for i in range(0, group_sum ,step):
            start_index = i
            end_index = min(step+i ,group_sum)
            batch_images = shuffled_img_paths[start_index * 81: end_index * 81]
            batch_json = []
            for j, p in  enumerate(batch_images):
                if j % 81 == 0:
                    key = p.split("\\")[-1].split("-")[1]
                    for entry in labels:
                     if entry.get('name') == key:
                        batch_json.append(entry)


            b_X, b_Y, _, _ = utils.generate_normalized_batch(batch_size=step,
                                                         batch_images=batch_images, batch_json=batch_json)
            np.savez(os.path.join(save_path, f"num_{i}.npz"), b_X=b_X, b_Y=b_Y)


def transfer_data():
      #xpreprocess.calculate_tr_avg_std('TRAIN')
      process_data()

if __name__ == "__main__":
    transfer_data()
