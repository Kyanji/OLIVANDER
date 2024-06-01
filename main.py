import argparse
import datetime
import os
import threading
import time

import numpy as np

import pickle
from libs.adv_step_mode import adv_step_mode
from libs.build_dnn import build_dnn
from libs.create_dataset import create_dataset
from libs.dataset_to_features import dataset_to_features
from libs.find_id import find_id
from libs.generate_conterfactual import generate_conterfactual

num_thread = 1
OPTIMIZED = False

parser = argparse.ArgumentParser(
    prog='OLIVANDER')

parser.add_argument("--mode", choices=['load_dataset', 'load_conterfactual', "generate_dataset"], required=True)
parser.add_argument('--pe_folder', default="/media/kyanji/data/dataset/pe-machine-learning-dataset/")
parser.add_argument('--conterfactual_path', default="conterfactuals.pickle")
parser.add_argument('--step', default=1000)
parser.add_argument('--section', default=1)
parser.add_argument('--iterative', default=0)
parser.add_argument('--iterative_on')
parser.add_argument('--c', default=100)
args = parser.parse_args()

pe_folder = args.pe_folder  # "/media/kyanji/data/dataset/pe-machine-learning-dataset/"
conterfactual_path = args.conterfactual_path  # "conterfactuals.pickle"

conf = args.mode
model = build_dnn()
model.load_weights("models/DNN_w.h5")

# CREATE DATASET FROM PE
if conf != "load_conterfactual":
    if conf == "generate_dataset":
        ds = create_dataset(pe_folder)
        x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta, x_test_meta = dataset_to_features(ds)
        del ds
    elif conf == "load_dataset":
        with open("pickle/dataset_lief.pickle", 'rb') as handle:
            ds = pickle.load(handle)
        x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta, x_test_meta = dataset_to_features(ds)
        del ds

    data = generate_conterfactual(x_train, x_val, x_test, y_train, y_val, y_test, model)
else:
    with open(conterfactual_path, "rb") as h:
        data = pickle.load(h)

    with open("pickle/dataset_lief.pickle", 'rb') as handle:
        ds = pickle.load(handle)
    x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta, x_test_meta = dataset_to_features(ds)
    del ds

mode = "OneShot"

STEPS = int(args.step)
SECTIONS = int(args.section)
if args.iterative == "1":
    if args.iterative_on == "section":
        mode = "IterOnSection"
    elif args.iterative_on == "step":
        mode = "IterOnStep"
    else:
        mode = "IterAll"

todo = list(data.keys())[:]
dir_output = "results/" + str(mode) + "_step" + str(STEPS) + "_sec" + str(SECTIONS) + "c" + str(args.c) + "/"
try:
    os.mkdir(dir_output)
except:
    pass
if mode == "OneShot":
    for d in os.listdir(dir_output):
        if ".json" in d:
            try:
                to_delete = int(d.split("-")[0])
                index = int(np.where(np.array(todo) == to_delete)[0][0])
                del todo[index]
            except:
                pass
threads = []
time_start = datetime.datetime.now()
while True:
    if len(threads) < num_thread:
        if len(todo) == 0:
            print("[+] DONE")
            break
        else:
            if len(todo) > 0:
                mi_n = todo[0]
                del todo[0]
                print("[+] STARTING\t" + str(mi_n))
                found, not_found, differences, differences_index, target, test, times = data[mi_n]
                index_file1 = find_id(test[0][0], x_test_meta)
                if ("temp-" + str(mi_n) + "-adv.exe") in dir_output:
                    resume = True
                else:
                    resume = False

                p1 = threading.Thread(target=adv_step_mode, args=(mode,
                                                                  differences_index, target, index_file1, model, mi_n,
                                                                  STEPS, dir_output, SECTIONS,
                                                                  pe_folder, False, OPTIMIZED, int(args.c)))

                threads.append(p1)
                p1.start()
            else:
                pass
    threads = [t for t in threads if t.is_alive()]
    time.sleep(5)

for t in threads:
    t.join()

time_end = datetime.datetime.now()
print((time_end - time_start).seconds)
f = open(dir_output + "time.txt", "w")
f.write(str((time_end - time_start).seconds))
f.close()
print("[+] ENDED")
