

dataset_name = "SMART"# Win5-LID SMART MPI-LFA


aug_factor = 'x8' # x32 x8
reduce_1 = True
no_wlbp=True
if dataset_name == "Win5-LID":
  # 9×9×434×625×3
  dataset_size=220
  to_size=(9*434, 9*434, 3)
  org_shape = (9,434,9,434,3)
  new_size = (7*434, 7*434, 3)
  new_shape = (7,434,7,434, 3)

elif dataset_name == "SMART":
  # 15×15×434×625×3
  dataset_size=1
  to_size=(9*400, 9*600, 3)
  org_shape = (9,400,9,600,3)
  new_size = (9*400, 9*600, 3)
  new_shape = (9,400,9,600, 3)

elif dataset_name == "MPI-LFA":
  dataset_size=336
  to_size = (15 * 434, 15 * 434, 3)
  # org_shape = (15,434,15,434,3)
  new_size = (7*434, 7*434, 3)
  new_shape = (7,434,7,434, 3)



from_shape = (9,400,9,600,3)
to_shape=(9*9,400,600,3)
n_tr_tt = dataset_size*8 # 1408 : 5632

tr_tt_ratio = 0.2
n_tt = int(n_tr_tt*tr_tt_ratio)
n_tr = n_tr_tt - n_tt
n_spa_feats =36
# gdd 1,8 wlbp 108,1
n_ang_feats_gdd = 8
n_ang_feats_wlbp = 108

img_format = "bmp" if dataset_name=="Win5-LID" else "png"
n_workers = 1
multip = False
group_number=81
batch_size = 1
app_data_raw_dir = './Datasets/APP/MINI_BATCH_DATA_TEST/'
app_data_npz_dir = './Datasets/APP/MINI_BATCH_NPZ/'
normalize_date_type = 'TRAIN'
weight_path = './WEIGHT/'

