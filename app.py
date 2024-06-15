import xmodels, constants
import os, glob, json
import pandas as pd
import numpy as np
import tensorflow as tf

import xpreprocess
from xtimer import Timer
from PIL import Image

dn = constants.dataset_name
img_format = constants.img_format
dataset_root, s = "./Datasets/", '/'
dn = constants.dataset_name
af = constants.aug_factor
r1 = constants.reduce_1
no_wlbp = constants.no_wlbp
img_format = constants.img_format

t = Timer()


def normalize(X, norm_method="stand"):
    print("g0")
    # save_dir =dataset_root+ dn+"-tensor-"+af+s
    if norm_method == "stand":
        save_path = './Datasets/APP/MEAN_STD/' + 'mean_std.npz'
    if not os.path.exists(save_path):
        print('The essential stats were not calculated!')
        return
    print("g1")
    with np.load(save_path) as tr_stats:
        if norm_method == "stand":
            print("g2")
            avg, std = tr_stats['avg'], tr_stats['std']
            print("g3")
            normalized_X = (X - avg) / (std + 1)
            print("g3")
    print("g4")

    return normalized_X


def generate_test_batches(batch_size=81, verbose=False):
    read_dir = constants.app_data_raw_dir
    save_dir = constants.app_data_npz_dir
    if verbose:
        t = Timer()
        t.start()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    read_img_paths = sorted(glob.glob(read_dir + "*." + img_format, recursive=False))
    read_img_paths = [pp.replace('\\', "/") for pp in read_img_paths]
    from_shape = constants.from_shape
    to_shape = constants.to_shape
    sam_labels_path = os.path.join(dataset_root, 'labels', 'DUT-LF_SAM_labels.json')
    with open(sam_labels_path, 'r') as file:
        labels = json.load(file)
    chunks = [read_img_paths[i:i + batch_size] for i in range(0, len(read_img_paths), batch_size)]
    for i, c in enumerate(chunks):
        #截取图片编号
        y_name=c[0].split(s)[-1].split("-")[1]
        print("yname{}".format(y_name))
        img_arrays = []
        l = len(c)
        b_X_shape = (1, to_shape[0], to_shape[1], to_shape[2], to_shape[3])
        b_X = np.zeros(b_X_shape, dtype=np.float32)
        b_y = np.zeros((1, l), dtype=np.float32)

        # 遍历 labels 并将 mos_array 添加到 b_y_list
        b_y_list = []
        for entry in labels:
            if entry['name'] == y_name:
                mos_values = entry.get("mos")
                if mos_values is not None:
                    mos_array = np.array(list(mos_values.values()), dtype=np.float32)
                    b_y_list.append(mos_array)

        # 将 b_y_list 中的数组按行连接
        if b_y_list:
            b_y = np.concatenate(b_y_list, axis=0)

        for j, p in enumerate(c):

            image = Image.open(p)
            data = np.asarray(image)
            img_arrays.append(data)

        img_arrays = np.array(img_arrays)
        print("q1")
        normalized_group_data=normalize(img_arrays)
        b_X_reshaped=normalized_group_data.reshape((1, constants.to_shape[0], constants.to_shape[1], constants.to_shape[2], constants.to_shape[3]))
        print("q2")
        np.savez(os.path.join(save_dir, f'{y_name}_batch_{i}.npz'), b_X=b_X_reshaped, b_y=b_y)
        save_paths = sorted(glob.glob(save_dir + "*.npz", recursive=False))
        save_paths = [pp.replace('\\', "/") for pp in save_paths]
        if verbose:
            t.lap()
            print(save_dir, "was saved")
    print('success! -', save_dir, 'were saved')
    if verbose:
        t.stop()
    return save_paths


def select_sai_range():
    n_sai = 101
    taget_n_sai = 49
    mid = n_sai // 2
    left = mid - taget_n_sai // 2
    right = mid + taget_n_sai // 2
    return left, right


def proc_sai(img):
    w, h = img.size
    w_offset = int((w - constants.to_size[0]) / 2)
    h_offset = int((h - constants.to_size[1]) / 2)
    l, r = w_offset, w_offset + constants.to_size[0]
    tp, b = h_offset, h_offset + constants.to_size[1]
    window = (l, tp, r, b)
    img2 = img.crop(window)
    return img2


def preproc_mpi_lfa(read_save_dir):
    read_img_paths = glob.glob(read_save_dir + "**/*." + img_format, recursive=True)
    read_img_paths = [pp.replace('\\', "/") for pp in read_img_paths]
    read_img_paths = sorted(read_img_paths)

    left, right = select_sai_range()

    # print(left,right)
    # print(right-left)

    new_shape = constants.new_shape
    # print(new_shape)
    lfi = np.zeros(new_shape)

    read_num = 0
    full_num_sai = 101
    while read_num < len(read_img_paths):

        sai_num = read_num % full_num_sai
        # print(sai_num)

        if sai_num == 100:
            # print("save sai")
            parts = img_path.split(s)
            name = f"{parts[-2]}.{img_format}"
            # print(name)
            # print(lfi.shape)
            # print(lfi)
            lfi = lfi.reshape(constants.new_size, order='F')
            # print(lfi.shape)
            lfi = Image.fromarray(lfi.astype(np.uint8))
            lfi.save(read_save_dir + "processed=" + name, img_format)

            lfi = np.zeros(new_shape)

        if not (left <= sai_num <= right):
            read_num += 1
            continue

        img_path = read_img_paths[read_num]

        img = Image.open(img_path)
        sai = proc_sai(img)
        data = np.asarray(sai)
        # print(i,j)

        u = (sai_num - left) // 7
        v = (sai_num - left) % 7
        lfi[u, :, v, :, :] = data

        # print(u,v)

        read_num += 1


# target shape: (7*434,7*434,3)
def mini_preprocess(save_dir):
    if dn == "MPI-LFA":
        preproc_mpi_lfa(save_dir)
        return save_dir

    read_dir = dataset_root + dn + "-tr-tt-x8" + s
    read_img_paths = glob.glob(read_dir + "*." + img_format, recursive=False)
    read_img_paths = [pp.replace('\\', "/") for pp in read_img_paths]

    # org_shape = (9,434,9,434,3)
    org_shape = constants.org_shape
    from_shape = constants.from_shape
    u_offset = int((org_shape[0] - from_shape[0]) / 2)
    v_offset = int((org_shape[2] - from_shape[2]) / 2)

    for p in read_img_paths:
        name = p.split(s)[-1]
        print('processing', name, '...')
        if "processed" in name:
            print("Already processed!")
            continue
        img = Image.open(p)

        # cut black
        if dn == "SMART":
            w, h = img.size
            l, r = 0, 625 * 15
            tp, b = 0, 434 * 15
            window = (l, tp, r, b)
            img = img.crop(window)

        # resize
        w, h = img.size
        # left, top, right, bottom
        w_offset = int((w - constants.to_size[0]) / 2)
        h_offset = int((h - constants.to_size[1]) / 2)
        l, r = w_offset, w_offset + constants.to_size[0]
        tp, b = h_offset, h_offset + constants.to_size[1]
        window = (l, tp, r, b)
        img2 = img.crop(window)

        # reduce_angular
        data = np.asarray(img2)
        a = data.reshape(org_shape, order='F')
        u_end, v_end = u_offset + from_shape[0], v_offset + from_shape[2]
        a = a[u_offset:u_end, :, v_offset:v_end, :, :]
        a = a.reshape(constants.new_size, order='F')
        img3 = Image.fromarray(a)

        img3.save(save_dir + "processed=" + name, img_format)
        t.lap()
    print('success! - mini preprocess')
    t.lap()
    return save_dir


def evaluate_tt(model_tuple, batch_size=10, normAL=True, verbose=False):
    t = Timer()
    t.start()
    model_path, model = model_tuple
    model_path = r"WEIGHT/DUTLF-V2.weights.h5"

    if model == None:
        if 'weight' in model_path:
            model = xmodels.get_Xmodel('LFACon', constants.to_shape, loss_weights=1)
            print("++----+")
            print(len(model.layers))
            model.load_weights(model_path)
        else:
            model = keras.ops.keras.models.load_model(model_path)
    if verbose:
        model.summary()
        print("generating test set...")
        t.lap()

    b_paths = generate_test_batches(batch_size=batch_size)
    if verbose:
        print("evaluating...")
        t.lap()

    loss_all, mae_all, mse_all = 0, 0, 0
    y_preds_mos, y_trues_mos, y_names = [], [], []
    model_outputs = {}
    # print(b_paths)
    # file_path = "./Datasets/SMART-tensor-x8/LFACon-tt-bz81-stand-nal1-nw1/tt_b0_bz81.npz"
    # data = np.load(file_path, allow_pickle=True)

    # # 打印文件中的所有内容
    # print("Contents of the .npz file:")
    # for key in data.keys():
    #   print(f"Key: {key}, Value: {data[key]}")
    for b_p in b_paths:

        with np.load(b_p, allow_pickle=True) as b:
            print(b_p)
            b_X = b["b_X"]
            b_Y = b["b_y"]
        # if model_struct == 'CNN_4D':
        #   b_X = b_X.reshape((batch_size, 81, 400, 600, 3), order='C')

        print("Shape of b_X:", b_X.shape)
        l = b_X.shape[0]
        y_pred = model.predict(b_X)
        y_name = b_p.split(s)[-1].split("-")[0]
        y_names.append(y_name)
        b_Y = b_Y.reshape(b_X.shape[0], 81)
        res_dict = model.evaluate(b_X, b_Y, verbose=verbose, batch_size=1, return_dict=True)
        # , use_multiprocessing=constants.multip, workers=constants.n_workers
        if "mae" in res_dict:
            loss, mae, mse = res_dict['loss'], res_dict['mae'], res_dict['loss']
        else:
            loss, mae, mse = res_dict['loss'], res_dict['loss'], res_dict['loss']
        y_pred_sqzd = np.squeeze(y_pred).tolist()
        b_y_sqzd = np.squeeze(b_Y).tolist()
        model_outputs[y_name] = {}
        # b_Y的预测值
        model_outputs[y_name]['y_pred'] = y_pred_sqzd
        #b_Y的真实值
        model_outputs[y_name]['y_true'] = b_y_sqzd
        if batch_size == 1 or "bz1." in b_p:
            y_preds_mos.append(y_pred_sqzd)
            y_trues_mos.append(b_y_sqzd)
            print('Predicted Score:', y_pred_sqzd)
        else:
            y_preds_mos += y_pred_sqzd
            y_trues_mos += b_y_sqzd
        loss_all += (loss * l)
        mae_all += (mae * l)
        mse_all += (mse * l)
        if verbose:
            t.lap()

    save_model_outputs(model_outputs, y_preds_mos, y_trues_mos, 'LFACon', s, y_names)
    print("evaluation result was saved.")
    if verbose:
        print("success! - evaluation")
    t.stop()
    return


def save_model_outputs(model_outputs, y_preds, y_trues, model_name, s, y_names):
    model_outputs_dir = "./Datasets/quality_predictions" + s
    if not os.path.exists(model_outputs_dir):
        os.makedirs(model_outputs_dir)
    model_outputs_path = model_outputs_dir + "y=" + model_name
    json.dump(model_outputs, open(model_outputs_path[:-3] + '.json', 'w'))
    print(y_names)
    return model_outputs_path


if __name__ == "__main__":
    print("frfr")

    t.start()

    model_tuple = ('./WEIGHT/times_0.weights.h5', None)


    save_dir = dataset_root+ dn+"-tr-tt-x8" + s
    # mini_preprocess(save_dir)
    # t.lap()

    tt_batch_save_dir = dataset_root+ dn+"-tensor-x8" + s
    if not os.path.exists(tt_batch_save_dir):
      os.makedirs(tt_batch_save_dir)

    batch_size = constants.batch_size
    normAL=True
    # verbose=False
    xpreprocess.calculate_tr_avg_std('APP')
    generate_test_batches(batch_size=81)
    print("success - generate_test_batches")
    # t.lap()

    evaluate_tt(model_tuple, batch_size=81, normAL=normAL,verbose=False)
    t.stop()