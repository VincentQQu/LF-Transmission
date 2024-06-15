import tensorflow as tf
from tensorflow.keras import backend as bkd
import gc

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import pandas as pd
import seaborn as sns
import os, sys, glob, pathlib, configparser, random, math, json, time
from xtimer import Timer
import matplotlib.pyplot as plt
import utils, xmodels, constants
import gc

bkd.set_floatx('float32')
input_shape = constants.to_shape
model_struct = 'LFACon'  # DADS_CNN, ALAS_DADS_CNN, ALASS_DADS_CNN, LFACon

##################################
version_no = 'v6.6'
##################################
fully_tained = True
# batch_size + eva_bz + n_val = 12
n_val, val_ratio, eva_bz = 2, 0.25, 1
n_epochs, n_mini_epochs = 2, 1
n_batch = int(n_epochs / n_mini_epochs)
dataset_root = ".\Datasets\\" if sys.platform == 'win32' else "./Datasets/"
dn = constants.dataset_name
af = constants.aug_factor
img_format = constants.img_format
##################################
epo_patience = 1
baseline = 100
##################################
monitor_val = True
saving_type = 'weight'  # model
reg_save_interv = 100
external_val = False
val_freq = 1  ###-

init_lr, end_lr, n_intv = 1e-6, 1e-6, 1  # 5-5 5-4 60e-6, 40e-6, 2
lr_change_intv, lr_repeat, lr_const_rmse = 200, True, 0  ###-
l_lrs = [round(lr, 6) for lr in np.geomspace(init_lr, end_lr, num=n_intv, endpoint=True).tolist()]  # linspace geomspace
##################################
start_lr_idx, lr_offset = 0, 0
##################################
no_wlbp = constants.no_wlbp
wts = [1, 0.01, 0.005, 0.005] if not no_wlbp else [1, 0.01, 0.01]  # [1,1,0.001,1]
loss_weights = {'mos': wts[0], "spatial": wts[1], "angular_gdd": wts[2], "angular_wlbp": wts[3]} if not no_wlbp else {
    'mos': wts[0], "spatial": wts[1], "angular_gdd": wts[2]}

nal = True
ver_dir = utils.get_version_dir(model_struct)
_, model_root, s = utils.root_paths()
shared_prex = [
    f"{model_struct}-[{version_no}]-{str(n_epochs)}epo{str(n_mini_epochs)}-{str(int(n_epochs / n_mini_epochs))}bsz2=weight-p",
    "--bst.json"]

#################################
shared_idx = ['6']
shared_paths = [shared_prex[0] + si + shared_prex[1] for si in shared_idx]
shared_paths = [ver_dir + sp for sp in shared_paths]

##################################
# startup_model_path = ver_dir+"LFACon-[v6.6]-MPI-LFA.h5"
startup_model_path = None
##################################


model_prex = ver_dir + f"{model_struct}-[{version_no}]-{n_epochs}epo{n_mini_epochs}-{n_batch}bsz{constants.batch_size}={saving_type}-p"
model_path, vn = utils.get_version_name(model_prex, ver_dir, '.json')


def build_train_model():
    print('Shared paths:', shared_paths)
    if model_struct == 'ALAS_DADS_CNN':
        historys = {"loss": [], "val_loss": [], "mos_loss": [], "val_mos_loss": [], "spatial_loss": [],
                    "val_spatial_loss": [], "angular_gdd_loss": [], "val_angular_gdd_loss": [], "angular_wlbp_loss": [],
                    "val_angular_wlbp_loss": []} if not no_wlbp else {"loss": [], "val_loss": [], "mos_loss": [],
                                                                      "val_mos_loss": [], "spatial_loss": [],
                                                                      "val_spatial_loss": [], "angular_gdd_loss": [],
                                                                      "val_angular_gdd_loss": []}
    else:
        historys = {"loss": [], "mae": [], "val_loss": [], "val_mae": []}

    bst_list = {}
    bst_path = model_path + '--bst.h5'
    real_time_model_path = model_path + '--rt.weights.h5'
    bst_paths = bst_path[:-3] + '.json'
    json.dump(bst_list, open(bst_paths, 'w'))
    print('lr list:', l_lrs)
    start_lr = l_lrs[start_lr_idx]
    # input_shape = (81,400,600,3)
    model = xmodels.get_Xmodel(model_struct, input_shape, fully_tained, start_lr, loss_weights=loss_weights)
    # if startup_model_path == None:
    #   model = xmodels.get_Xmodel(model_struct, input_shape, fully_tained,start_lr, loss_weights=loss_weights)
    # else:
    #   assert startup_model_path.split(s)[-1].split('-')[0]==model_struct
    #   model = xmodels.get_Xmodel(model_struct, input_shape, fully_tained, start_lr, loss_weights=loss_weights)
    #   model.load_weights(startup_model_path)
    startup_model_path = "./WEIGHT/times_2.weights.h5"
    # 检查是否有启动模型路径，如果有，则从该路径加载权重
    if startup_model_path:
        model.load_weights(startup_model_path)
        print("Loaded weights from:", startup_model_path)
    else:
        print("No startup model path provided. Training from scratch.")

    print(model)
    print("获得初始权重 produce good work")

    monitor = 'loss'
    if monitor_val:
        monitor = 'val_loss'

    es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=epo_patience, verbose=2, restore_best_weights=True)
    csvlg = tf.keras.callbacks.CSVLogger("Datasets/history/training_history.csv", separator=",", append=True)
    tnan = tf.keras.callbacks.TerminateOnNaN()
    cbks = [es, csvlg, tnan]
    # reduce_lr = ReduceLROnPlateau(monitor=monitor,factor=0.5,patience=5, min_lr=0.001)
    b_save_prex = model_path + '--bch_.h5'
    print('start training...')
    b_X = b_y = X_val = y_val = None
    # list_with_replace_tr, list_with_replace_val = [], []

    # 加载验证数据
    val_data_path = "./Datasets/COMPRESS-DATA-VAL/*.npz"
    val_data_files = sorted(glob.glob(val_data_path))
    random.shuffle(val_data_files)  # 初始打乱验证文件列表

    val_file_index = 0  # 验证文件索引
    val_data = np.load(val_data_files[val_file_index])
    X_val = val_data['b_X']
    Y_val = val_data['b_Y']
    val_example_index = 0  # 单个文件中的样本索引

    old_lr = start_lr
    # model.save_weights(real_time_model_path)

    for i in range(2):
        b_n = i + 1
        print('-' * 50, 'Bacth_' + str(b_n), '-' * 50)
        # b_save_path = b_save_prex[:-3] + str(b_n)+'.h5'
        # print('lr:', model.optimizer.get_config())
        new_lr_idx = int((i + lr_offset) / lr_change_intv % n_intv) if lr_repeat else int(
            (i + lr_offset) / lr_change_intv)
        new_lr = l_lrs[min(start_lr_idx + new_lr_idx, len(l_lrs) - 1)]
        if new_lr != old_lr:
            bkd.set_value(model.optimizer.learning_rate, new_lr)

            print('new lr:', new_lr)
            old_lr = new_lr
        tr_type = "train"


        # 循环进行训练 加载图片和json
        read_dir = "./Datasets/COMPRESS-DATA/"
        # if list_with_replace == None:
        read_data_paths = sorted(glob.glob(read_dir + "*." + "npz", recursive=False))
        random.shuffle(read_data_paths)  # Shuffle paths each iteration
        X_val = []
        for u, data_path in enumerate(read_data_paths):
            print('=' * 50, 'Step_' + str(u), '-' * 50)

            with np.load(data_path) as data:
                b_X = data['b_X']
                b_Y = data['b_Y']
            print("b_X:{},b_Y:{}".format(b_X.shape, b_Y.shape))

            # 加载一个验证样本

            if val_example_index >= len(X_val):  # 如果当前文件的样本已经用完，加载下一个文件
                val_file_index += 1

                if val_file_index >= len(val_data_files):  # 如果所有文件都已经使用过，重新开始
                    random.shuffle(val_data_files)
                    val_file_index = 0
                val_data = np.load(val_data_files[val_file_index])
                X_val = val_data['b_X']
                Y_val = val_data['b_Y']
                val_example_index = 0  # 重置样本索引

            current_X_val = X_val[val_example_index:val_example_index + 1]
            current_Y_val = Y_val[val_example_index * 81:val_example_index * 81 + 81]
            current_Y_val = current_Y_val.reshape(1, 81)  # 调整形状以适应模型
            val_example_index += 1  # 移动到下一个样本

            b_Y = b_Y.reshape(b_X.shape[0], 81)

            # 执行训练并验证
            h = model.fit(b_X, b_Y, batch_size=4, epochs=1, verbose=1, validation_data=(current_X_val, current_Y_val),callbacks=cbks)

            del h
            h = None  # 显式地将h设置为None，帮助释放内存

            b_save_path = constants.weight_path + 'times_' + str(i) + ".weights.h5"
            if saving_type == 'weight':
                model.save_weights(b_save_path)
            else:
                model.save(b_save_path)

            print(b_n, "out of", n_batch, "batches completed")


            # historys = merge_save_history(historys, h.history)
            model.save_weights(real_time_model_path)

            del b_X, b_Y, current_X_val, current_Y_val
            gc.collect()

            t.lap()
        tf.keras.backend.clear_session()
        del data, val_data, X_val, Y_val
        gc.collect()


    print("over")
    return model, historys


def merge_save_history(hs, h):
    l = len(h['loss'])
    for k, v in h.items():
        if len(v) != l:
            hs[k] += ([0] * l)
        else:
            hs[k] += v
    json.dump(hs, open("./Datasets/history/training_history" + '.json', 'w'))
    return hs


def save_pic_results(historys):
    mse_label = 'mse' if model_struct == "ALAS_DADS_CNN" else 'loss'
    plt.plot(np.sqrt(historys[mse_label]), label='rmse')
    plt.plot(np.sqrt(historys['val_' + mse_label]), label='val_rmse')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.ylim([0, 10])
    plt.legend(loc='lower right')

    last_n = 100
    print('=' * 10, 'Training Results', '=' * 10)
    tr_rmse = np.mean(np.sqrt(historys[mse_label])[-last_n:])
    print(f"Mean RMSE of last {last_n} epoches: {tr_rmse}")
    print('=' * 10, 'Validation Results', '=' * 10)
    val_rmse = np.mean(np.sqrt(historys['val_' + mse_label])[-last_n:])
    print(f"Mean RMSE of last {last_n} epoches: {val_rmse}")
    print('=' * 10, 'Test Results', '=' * 10)
    # 评估模型质量
    model_tuple = ('./WEIGHT/times_0.weights.h5', None)
    normAL = True
    loss, mae, mse, rmse, srcc, lcc = utils.evaluate_tt(model_tuple, batch_size=81, normAL=normAL, verbose=False)
    print("-" * 10, 'loss: {:5.4f}, mae: {:5.4f}, mse: {:5.4f}'.format(loss, mae, mse), "-" * 10)
    print("+" * 10, "TEST RMSE: {:5.4f}".format(rmse), "+" * 10, '\n')
    total_time = int(t.total_t())
    plt.title(
        'RMSE tr:{:.4f}val:{:.4f}tt:{:.4f} {}s SRCC:{:.4f}LCC:{:.4f}'.format(tr_rmse, val_rmse, rmse, total_time, srcc,
                                                                             lcc))
    plt.savefig(model_path + '.png', dpi=1200)  # os:{} ,sys.platform


if __name__ == "__main__":
    # main
    print('=' * 50)
    t = Timer()
    t.start()
    # if len(sys.argv) > 1:
    #   if sys.argv[1][0] == 'd':
    #     if sys.argv[1] == 'dcsc':
    #       strategy = tf.distribute.experimental.CentralStorageStrategy()
    #       # exit(), compute_devices=['/job:localhost/replica:0/task:0/device:GPU:1'], parameter_device=None
    #     elif sys.argv[1] == 'dm':
    #       strategy = tf.distribute.MirroredStrategy()
    #       # cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    #     elif sys.argv[1] == 'dcsg':
    #       strategy = tf.distribute.experimental.CentralStorageStrategy(parameter_device='/job:localhost/replica:0/task:0/device:GPU:0')
    #     elif sys.argv[1] == 'dmh':
    #       strategy = tf.distribute.MirroredStrategy(devices=['/job:localhost/replica:0/task:0/device:GPU:0','/job:localhost/replica:0/task:0/device:GPU:1'],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    #     else:
    #       strategy = tf.distribute.experimental.CentralStorageStrategy()
    #     # train with the strategy
    #     with strategy.scope():
    #       model, historys = build_train_model()
    #   elif sys.argv[1][0] == 'g':
    #     # tf.config.set_soft_device_placement(True)
    #     gpu_prex = '/job:localhost/replica:0/task:0/device:GPU:'
    #     with tf.device('/device:CPU:0'):
    #       with tf.device(gpu_prex+sys.argv[1][1]):
    #         model, historys = build_train_model()
    #   elif sys.argv[1][0] == 'c':
    #     with tf.device('/device:CPU:0'):
    #       model, historys = build_train_model()
    # else:
    #   model, historys = build_train_model()

    with tf.device('/device:gpu:0'):
        model, historys = build_train_model()
    if saving_type == 'weight':
        model.save_weights(model_path + ".weights" + ".h5")
    else:
        model.save(model_path + '.h5')

    # history_dict_list = json.load(open(model_path+'.json', 'r'))
    # save_pic_results(historys)
    # save_pic_results(historys)
    print('=' * 100)
    print('=' * 100)
    t.stop()
