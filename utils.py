import numpy as np
# from sklearn.model_selection import train_test_split
import os, glob, sys, constants, random, xmodels, json, math
from xtimer import Timer
import tensorflow as tf
from PIL import Image
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from csv import writer
dn = constants.dataset_name
af = constants.aug_factor
r1 = constants.reduce_1
no_wlbp = constants.no_wlbp
img_format = constants.img_format
split_char = '-'
# lcc_test, _ = pearsonr([1,1,1],[1,1,1])

dataset_root, s = "./Datasets/", '/'



def generate_normalized_batch(model_struct='DADS_CNN', batch_size=81, batch_images=None, batch_json=None,
                              tr_or_tt="train", list_with_replace=None, norm_method="stand", normAL=True,
                              verbose=True):
  if verbose:
    print("generating batch...")
  dataset_root, _, s = root_paths()
  labels = batch_json
  to_shape = constants.to_shape
  b_X = np.zeros((batch_size*to_shape[0], to_shape[1], to_shape[2], to_shape[3]), dtype=np.float32)
  b_X_shape_new = np.zeros((batch_size, to_shape[0], to_shape[1], to_shape[2], to_shape[3]), dtype=np.float32)
  b_y_list = []

  for entry in batch_json:
    mos_values = entry.get("mos")
    if mos_values is not None:
      mos_array = np.array(list(mos_values.values()), dtype=np.float32)
      b_y_list.append(mos_array)

  if b_y_list:
    b_y = np.concatenate(b_y_list)
  else:
    b_y = np.zeros((batch_size, 1), dtype=np.float32)

  for i in range(0, len(batch_images), 81):
    group_images = batch_images[i:i + 81]
    group_data = []

    for p in group_images:
      image = Image.open(p)
      data = np.asarray(image)
      image.close()
      group_data.append(data)
    group_data = np.array(group_data)
    normalized_group_data = normalize(group_data, norm_method)
    b_X[i:i + 81] = normalized_group_data
  b_X_reshaped = b_X.reshape((batch_size, to_shape[0], to_shape[1], to_shape[2], to_shape[3]))
  return b_X_reshaped, b_y, list_with_replace, batch_images


def evaluate_tt(model_tuple, batch_size=10, normAL=True, verbose=False):
  t = Timer()
  t.start()
  model_path, model = model_tuple

  if model == None:
    if 'weight' in model_path:
      model = xmodels.get_Xmodel('LFACon', constants.to_shape, loss_weights=1)
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

  for b_p in b_paths:
    with np.load(b_p, allow_pickle=True) as b:
      b_X = b["b_X"]
      b_Y = b["b_y"]
    # if model_struct == 'CNN_4D':
    #   b_X = b_X.reshape((batch_size, 81, 400, 600, 3), order='C')

    l = b_X.shape[0]
    y_pred = model.predict(b_X)
    y_name = b_p.split(s)[-1].split("-")[0]
    y_names.append(y_name)
    b_Y = b_Y.reshape(b_X.shape[0], 81)
    res_dict = model.evaluate(b_X, b_Y, verbose=verbose, batch_size=1, return_dict=True)

    if "mae" in res_dict:
      loss, mae, mse = res_dict['loss'], res_dict['mae'], res_dict['loss']
    else:
      loss, mae, mse = res_dict['loss'], res_dict['loss'], res_dict['loss']
    y_pred_sqzd = np.squeeze(y_pred).tolist()
    b_y_sqzd = np.squeeze(b_Y).tolist()
    model_outputs[y_name] = {}
    model_outputs[y_name]['y_pred'] = y_pred_sqzd
    model_outputs[y_name]['y_true'] = b_y_sqzd
    if batch_size == 1 or "bz1." in b_p:
      y_preds_mos.append(y_pred_sqzd)
      y_trues_mos.append(b_y_sqzd)
      # print('Predicted Score:', y_pred_sqzd)
    else:
      y_preds_mos += y_pred_sqzd
      y_trues_mos += b_y_sqzd
    loss_all += (loss * l)
    mae_all += (mae * l)
    mse_all += (mse * l)
    if verbose:
      t.lap()

  n_tt = constants.n_tt
  loss_all = loss_all/n_tt
  mae_all = mae_all/n_tt
  mse_all = mse_all/n_tt
  rmse = np.sqrt(mse_all)
  srcc, _ = spearmanr(y_preds_mos,y_trues_mos)
  _, _ = pearsonr(y_preds_mos,y_trues_mos)
  lcc, _ = pearsonr(y_preds_mos,y_trues_mos)
  show_evaluation_results(loss_all, mae_all, mse_all, rmse, srcc, lcc)
  new_row = ["LFACon", loss_all,mae_all,mse_all,rmse,srcc,lcc]
  df = pd.DataFrame([new_row], columns=["Model","Loss", "MAE", "MSE", "RMSE", "SRCC", "LCC"])

  save_model_outputs(model_outputs, y_preds_mos, y_trues_mos, 'LFACon', s, y_names)
  print("evaluation result was saved.")
  if verbose:
    print("success! - evaluation")
  t.stop()
  return loss_all, mae_all, mse_all, rmse, srcc, lcc


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
    y_name = c[0].split(s)[-1].split("-")[1]
    img_arrays = []
    l = len(c)
    b_X_shape = (1, to_shape[0], to_shape[1], to_shape[2], to_shape[3])
    b_X = np.zeros(b_X_shape, dtype=np.float32)
    b_y = np.zeros((1, l), dtype=np.float32)

    b_y_list = []
    for entry in labels:
      if entry['name'] == y_name:
        mos_values = entry.get("mos")
        if mos_values is not None:
          mos_array = np.array(list(mos_values.values()), dtype=np.float32)
          b_y_list.append(mos_array)

    if b_y_list:
      b_y = np.concatenate(b_y_list, axis=0)

    for j, p in enumerate(c):
      image = Image.open(p)
      data = np.asarray(image)
      img_arrays.append(data)

    img_arrays = np.array(img_arrays)
    normalized_group_data = normalize(img_arrays)
    b_X_reshaped = normalized_group_data.reshape(
      (1, constants.to_shape[0], constants.to_shape[1], constants.to_shape[2], constants.to_shape[3]))
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




def append_list_as_row(fn, l):
  with open(fn, 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(l)



def hasnan(X,model_struct):
  return False
  if model_struct!="ALAS_DADS_CNN":
    return np.any(np.isnan(X))
  for _,v in X.items():
    # print(np.isnan(np.sum(v)), np.any(np.isnan(v)))
    if np.any(np.isnan(v)):
      return True
  return False



def sqz_y_pred(y_pred):
  y_pred_sqzd={}
  y_pred_sqzd['mos'] = np.squeeze(y_pred[0]).tolist()
  y_pred_sqzd['spatial'] = np.squeeze(y_pred[1]).tolist()
  y_pred_sqzd['angular_gdd'] = np.squeeze(y_pred[2]).tolist()
  if not no_wlbp:
    y_pred_sqzd['angular_wlbp'] = np.squeeze(y_pred[3]).tolist()
  return y_pred_sqzd



def sqz_b_y(b_y):
  b_y_sqzd={}
  for k in b_y.keys():
    b_y_sqzd[k] = np.squeeze(b_y[k]).tolist()
  return b_y_sqzd



def scale_spatial(spatial):
  spa_min,spa_max = 25.811637432352,65.8772872416632
  return (spa_max-spatial)/(spa_max-spa_min)*4+1



def show_evaluation_results(loss_all, mae_all, mse_all, rmse, srcc, lcc):
  print('-'*10,'Evaluation Results','-'*10)
  print('-'*10,'loss: {:.4f} MAE: {:.4f} MSE: {:.4f}'.format(loss_all,mae_all,mse_all),'-'*10)
  print('+'*10,'RMSE: {:.4f} SRCC: {:.4f} LCC: {:.4f}'.format(rmse,srcc,lcc),'+'*10)
  return loss_all, mae_all, mse_all, rmse, srcc, lcc



def save_model_outputs(model_outputs,y_preds, y_trues, model_name, s, y_names):
  model_outputs_dir = get_version_dir()+"model_outputs"+s
  if not os.path.exists(model_outputs_dir):
    os.makedirs(model_outputs_dir)
  model_outputs_path=model_outputs_dir+"y="+model_name
  json.dump(model_outputs, open(model_outputs_path[:-3]+'.json', 'w'))
  if len(y_trues) == len(y_names):
    diffs = []
    for _,yp,yt in zip(y_names,y_preds,y_trues):
      diffs.append(yp-yt)
    data = {"y_name":y_names,"y_pred":y_preds,"y_true":y_trues,"diff":diffs}
    df = pd.DataFrame(data, columns=["y_name","y_pred","y_true","diff"])
    df.to_csv(model_outputs_path+".csv")
  return model_outputs_path



def normalize(X, norm_method="stand"):
  dataset_root, _, s = root_paths()
  save_dir =dataset_root+ dn+"-tensor-"+af+s
  if norm_method=="stand":
    save_path = './Datasets/mean_std/SMART/mean_std.npz'
  else:
    save_path = save_dir+'min_max.npz'
  if not os.path.exists(save_path):
    print('The essential stats were not calculated!')
    return
  with np.load(save_path) as tr_stats:
    if norm_method=="stand":
      avg, std = tr_stats['avg'], tr_stats['std']
      normalized_X = (X-avg)/(std+1)
    else:
      min_a, max_a = tr_stats['min_a'], tr_stats['max_a']
      normalized_X = (X-min_a)/(max_a-min_a)
  return normalized_X



def get_version_paths(model_struct=None):
  print(get_version_dir(model_struct))
  _, _,s = root_paths()
  return glob.glob(get_version_dir(model_struct)+f'**{s}*.h5', recursive=True)



def get_version_dir(model_struct=None):
  _, model_root,s = root_paths()
  ver_dir = model_root+f"versions-{dn}{s}"
  if model_struct != None:
    ver_dir = model_root+f"versions-{dn}{s}{model_struct}{s}"
  if not os.path.exists(ver_dir):
    os.makedirs(ver_dir)
  return ver_dir



def root_paths():
  dataset_root = ".\Datasets\\" if sys.platform =='win32' else "./Datasets/"
  model_root = ".\\" if sys.platform =='win32' else "./"
  seperator = "\\" if sys.platform =='win32' else "/"
  return dataset_root, model_root, seperator



def kws_in_str(kws,st, strict):
  for kw in kws:
    if not kw in st and strict:
      return False
    if kw in st  and not strict:
      return True
  return True



def get_version_name(model_prex, ver_dir, file_type):
    n=0
    for i in range(1000):
      n=i
      file_name =model_prex +str(n)+file_type
      # keywords.append(file_type)
      versions_dirs = glob.glob(ver_dir+'*'+file_type)
      # n_used = len([d for d in versions_dirs if kws_in_str(keywords, d, True)] )>0
      # if not n_used:
      #   break
      if not file_name in versions_dirs:
        break
    model_save_dir=model_prex+str(n)
    return model_save_dir, n

def find_largest_i_file(directory, prefix='times_'):
    max_i = -1
    max_file = None
    for file in os.listdir(directory):
        if file.startswith(prefix) and file.endswith(".weights.h5"):
            try:
                i = int(file[len(prefix):-len(".weights.h5")])
                if i > max_i:
                    max_i = i
                    max_file = os.path.join(directory, file)
            except ValueError:
                pass  # Ignore files that don't match expected format
    return max_file




def sort_batch_dirs(batch_dirs):
  return sorted(batch_dirs, key=lambda bn: int(bn[bn.find('_batch')+6:-4]))




if __name__ == "__main__":
  pass
