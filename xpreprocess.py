import time

from PIL import Image, ImageChops, ImageEnhance, ImageOps
import numpy as np
from xtimer import Timer
import sys, glob, os, random
import pandas as pd
import utils, constants
import threading
import multiprocessing
from multiprocessing import Process, Pool
print_lock = threading.Lock()
# 773s before avg_std; after avg(479s) 1252s; after std(633s) 1885s;
t = Timer()
t.start()
exclude_ref = True
dataset_root, _, s = utils.root_paths()
dn = constants.dataset_name
af = constants.aug_factor ###-
img_format = constants.img_format



def flatten_dataset(save_dir):
  # \Datasets\MPI-LFA\
  read_dir = dataset_root+dn+s
  read_img_paths = glob.glob(read_dir+"**/*."+img_format, recursive=True)
  ref_word = 'Reference'
  if dn=="SMART":
    ref_word='SRCs'
  for p in read_img_paths:
    if exclude_ref:
      if ref_word in p:
        continue
    parts = p.split(s)
    name = parts[-3]+'-'+parts[-2]+'-'+parts[-1]
    print(save_dir+name)
    os.rename(p, save_dir+name)
  print('success! - flatten_dataset')
  if dn=="SMART":
    cut_black()
  t.lap()
  return save_dir



def trim():
  read_dir = dataset_root+ dn+"-flatten" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  
  for p in read_img_paths:
    name = p.split(s)[-1]
    print('processing',name,'...')
    img = Image.open(p)
    w, h = img.size
    bg = Image.new(img.mode, img.size, img.getpixel((w-1,h-1)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    # l,r = 0, 625*15
    # tp,b = 0, 434*15
    # window = (l,tp,r,b)
    window=bbox
    new_img = img.crop(window)
    new_img.save(read_dir+name,img_format)
    t.lap()
  print('success! - trim')
  t.lap()
  return read_dir



def cut_black():
  read_dir = dataset_root+ dn+"-flatten" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  
  for p in read_img_paths:
    name = p.split(s)[-1]
    print('processing',name,'...')
    img = Image.open(p)
    w, h = img.size
    l,r = 0, 625*15
    tp,b = 0, 434*15
    window = (l,tp,r,b)
    new_img = img.crop(window)
    new_img.save(read_dir+name,img_format)
    t.lap()
  print('success! - cut_black')
  t.lap()
  return read_dir



def resize(save_dir):
  read_dir = dataset_root+ dn+"-flatten" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  for p in read_img_paths:
    name = p.split(s)[-1]
    print('processing',name,'...')
    img = Image.open(p)
    w, h = img.size
    # left, top, right, bottom
    w_offset = int((w-constants.to_size[0])/2)
    h_offset = int((h-constants.to_size[1])/2)
    l,r = w_offset, w_offset+constants.to_size[0]
    tp,b = h_offset, h_offset+constants.to_size[1]
    window = (l,tp,r,b)
    new_img = img.crop(window)
    new_img.save(save_dir+name,img_format)
    t.lap()
  print('success! - resize')
  t.lap()
  return save_dir



def reduce_angular(save_dir):
  read_dir = dataset_root+ dn+"-resized" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  # (15,434,15,434,3)
  org_shape = constants.org_shape
  # (7,434,7,434,3)
  from_shape = constants.from_shape
  u_offset=int((org_shape[0]-from_shape[0])/2)
  v_offset=int((org_shape[2]-from_shape[2])/2)
  print(u_offset,v_offset)
  for p in read_img_paths:
    name = p.split(s)[-1]
    image = Image.open(p)
    data = np.asarray(image)
    a = data.reshape(org_shape, order='F')
    u_end, v_end =u_offset+from_shape[0], v_offset+from_shape[2]
    a = a[u_offset:u_end,:,v_offset:v_end,:,:]
    a = a.reshape(constants.new_size, order='F')
    new_img = Image.fromarray(a)
    new_img.save(save_dir+name,img_format)
    t.lap()
  print('success! - reduce_angular')
  t.lap()
  return save_dir


# version 1.0
# def rotate_flip(save_dir):
#   # read_dir = dataset_root+ dn+"-reduce-angular" + s
#   read_dir = dataset_root + dn + "-flatten" + s
#   read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
#   rotate_angles = [0,180]
#   for p in read_img_paths:
#     name = p.split(s)[-1]
#     print('processing',name,'...')
#     image = Image.open(p)
#     for r in rotate_angles:
#       # 旋转图片
#       img = image.rotate(r)
#       # name+rotate+rotate_angular+.png
#       new_name = name[:-4]+'+rotate'+str(r)+name[-4:]
#       img.save(save_dir+new_name,img_format)
#       # 水平翻转
#       img_f = img.transpose(Image.FLIP_LEFT_RIGHT)
#       new_name_f = new_name[:-4]+'+lrflip'+new_name[-4:]
#       img_f.save(save_dir+new_name_f,img_format)
#   print('success! - rotate_flip')
#   t.lap()
#   return save_dir


# version 1.3
# def rotate_flip(save_dir):
#   # read_dir = dataset_root+ dn+"-reduce-angular" + s
#   read_dir = dataset_root + dn + "-flatten" + s
#   read_img_paths = glob.glob(read_dir + "*." + img_format, recursive=False)
#   rotate_angles = [0, 180]
#   color_factors = [0.8, 1, 1.2]
#   gamma_values = [0.8, 1.0, 1.2]
#   # noise_levels = [0.1, 0.2, 0.3]  # 随机噪声水平
#   noise_levels = [0.1]  # 随机噪声水平
#   for p in read_img_paths:
#     name = p.split(s)[-1]
#     print('processing', name, '...')
#     image = Image.open(p)
#     for r in rotate_angles:
#       # 旋转图片
#       img = image.rotate(r)
#       # name+rotate+rotate_angular+.png
#       new_name = name[:-6] + 'rotate-' + str(r) + name[-7:-4] + name[-4:]
#       # 水平翻转
#       img_f = img.transpose(Image.FLIP_LEFT_RIGHT)
#       new_name_f = new_name[:-6] + 'lrflip' + name[-7:-4]  + new_name[-4:]
#
#       # 颜色缩放和gamma校正
#       for factor in color_factors:
#         enhancer = ImageEnhance.Color(img)
#         img_color_scaled = enhancer.enhance(factor)
#         for gamma in gamma_values:
#           img_gamma = ImageEnhance.Brightness(img_color_scaled).enhance(gamma)
#           new_name_combined = name[:-6] + 'color' + str(factor) + '-gamma' + str(gamma) + name[-7:-4]  + new_name[-4:]
#           save_path_combined = os.path.join(save_dir, new_name_combined)
#           if not os.path.exists(save_path_combined):
#             img_gamma.save(save_path_combined, img_format)
#             # 添加随机噪声
#             for noise_level in noise_levels:
#               noisy_img = add_noise(img_gamma, noise_level)
#               noisy_name = name[:-6] + f'color{str(factor)}-gamma{str(gamma)}-noise{int(noise_level * 10)}' + name[-7:-4]  + new_name[-4:]
#               noisy_img.save(os.path.join(save_dir, noisy_name), img_format)
#
#       for factor in color_factors:
#         enhancer = ImageEnhance.Color(img_f)
#         img_color_scaled = enhancer.enhance(factor)
#         for gamma in gamma_values:
#           img_gamma = ImageEnhance.Brightness(img_color_scaled).enhance(gamma)
#           new_name_combined = new_name_f[:-6] + 'color' + str(factor) + '-gamma' + str(gamma) + name[-7:-4]  + new_name[-4:]
#           save_path_combined = os.path.join(save_dir, new_name_combined)
#           if not os.path.exists(save_path_combined):
#             img_gamma.save(save_path_combined, img_format)
#           for noise_level in noise_levels:
#             noisy_img = add_noise(img_gamma, noise_level)
#             noisy_name_f = new_name_f[:-6] + f'color{str(factor)}-gamma{str(gamma)}-noise{int(noise_level * 10)}' + name[-7:-4]  + new_name_f[-4:]
#             noisy_img.save(os.path.join(save_dir, noisy_name_f), img_format)
#
#   print('success! - rotate_flip')
#   t.lap()
#   return save_dir

def rotate_flip_single(p):
  save_dir = dataset_root+ dn+"-rotated-flipped" + s
  name = p.split(s)[-1]
  print('processing', name, '...')
  image = Image.open(p)
  # rotate_angles
  for r in [0, 180]:
  # 旋转图片
      img = image.rotate(r)
      # name+rotate+rotate_angular+.png
      new_name = name[:-6] + 'rotate' + str(r) + name[-7:]
      img.save(os.path.join(save_dir, new_name))

      # 水平翻转
      img_f = img.transpose(Image.FLIP_LEFT_RIGHT)
      new_name_f = new_name[:-6] + 'lrflip' + new_name[-7:]
      img_f.save(os.path.join(save_dir, new_name_f))

      # 添加噪声
      noisy_img = add_noise(img)
      noisy_name = new_name[:-6] + 'noise1' + new_name[-7:]
      noisy_img.save(os.path.join(save_dir, noisy_name))

      noisy_img_f = add_noise(img_f)
      noisy_name_f = new_name_f[:-6] + 'noise1' + new_name_f[-7:]
      noisy_img_f.save(os.path.join(save_dir, noisy_name_f))
  pass

def rotate_flip(save_dir):
  read_dir = dataset_root + dn + "-flatten" + s
  read_img_paths = glob.glob(read_dir + "*." + img_format, recursive=False)

  with Pool() as pool:
    pool.map(rotate_flip_single, read_img_paths)

  print('success! - rotate_flip')
  t.lap()
  return save_dir


def add_noise(image):
  """向图像添加随机噪声"""
  if image.mode != 'RGB':
    image = image.convert('RGB')

  width, height = image.size
  noisy_image = Image.new("RGB", (width, height))
  for x in range(width):
    for y in range(height):
      r, g, b = image.getpixel((x, y))
      noise = random.randint(-int(255 * 0.1), int(255 * 0.1))
      r = max(0, min(255, r + noise))
      g = max(0, min(255, g + noise))
      b = max(0, min(255, b + noise))
      noisy_image.putpixel((x, y), (r, g, b))
  return noisy_image


def split_list(a_list, num_splits):
  """Split a list into approximately equal sized chunks"""
  avg = len(a_list) / float(num_splits)
  out = []
  last = 0.0

  while last < len(a_list):
    out.append(a_list[int(last):int(last + avg)])
    last += avg

  return out


def process_images_in_parallel(save_dir, read_img_paths, num_processes):
  chunks = split_list(read_img_paths, num_processes)
  processes = []
  for chunk in chunks:
    p = Process(target=rotate_flip, args=(save_dir, chunk))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()



# x8 -------------------------------------------------
# def process_image(p, save_dir):
#   name = os.path.basename(p)
#   print('processing', name, '...')
#   image = Image.open(p)
#   rotate_angles = [0, 180]
#   for r in rotate_angles:
#     # 旋转图片
#     img = image.rotate(r)
#     # name+rotate+rotate_angular+.png
#     new_name = name[:-6] + 'rotate' + str(r) + name[-7:]
#     img.save(os.path.join(save_dir, new_name))
#
#     # 水平翻转
#     img_f = img.transpose(Image.FLIP_LEFT_RIGHT)
#     new_name_f = new_name[:-6] + 'lrflip' + new_name[-7:]
#     img_f.save(os.path.join(save_dir, new_name_f))
#
#     # 添加噪声
#     noisy_img = add_noise(img)
#     noisy_name = new_name[:-6] + 'noise1' + new_name[-7:]
#     noisy_img.save(os.path.join(save_dir, noisy_name))
#
#     noisy_img_f = add_noise(img_f)
#     noisy_name_f = new_name_f[:-6] + 'noise1' + new_name_f[-7:]
#     noisy_img_f.save(os.path.join(save_dir, noisy_name_f))
#   print('Finished processing', name)
#
#
# def rotate_flip_multi_thread(save_dir):
#   read_dir = dataset_root + dn + "-flatten" + s
#   read_img_paths = glob.glob(read_dir + "*." + img_format, recursive=False)
#
#   # 设置线程池大小，可以根据实际情况调整
#   max_threads = 20
#   threads = []
#   for p in read_img_paths:
#     thread = threading.Thread(target=process_image, args=(p, save_dir))
#     threads.append(thread)
#     if len(threads) >= max_threads:
#       # 启动线程
#       for t in threads:
#         t.start()
#       # 等待所有线程完成
#       for t in threads:
#         t.join()
#       threads = []
#
#   # 启动并等待剩余的线程
#   for t in threads:
#     t.start()
#   for t in threads:
#     t.join()
#
#   print('Success! - rotate_flip')
# x8---------------------------------------------------------







def crop(save_dir):
  read_dir = dataset_root+ dn+"-rotated-flipped" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  # 0, 1, 2, 3: 00,01,10,11
  h,w=constants.new_size[0],constants.new_size[1]
  # left, top, right, bottom
  crop_windows = [(0,0,int(w/2),int(h/2)), (int(w/2),0,w,int(h/2)), (0,int(h/2),int(w/2),h), (int(w/2),int(h/2),w,h)]
  for p in read_img_paths:
    name = p.split(s)[-1]
    print('processing',name,'...')
    image = Image.open(p)
    for i, cw in enumerate(crop_windows):
      img = image.crop(cw)
      new_name = name[:-4]+'+crop'+str(i)+name[-4:]
      img.save(save_dir+new_name,img_format)
  print('success! - crop')
  t.lap()
  return save_dir



def tr_tt_split(save_dir):
  if af =='x32':
    read_dir = dataset_root+ dn+"-crop" + s
  else:
    read_dir = dataset_root+ dn+"-rotated-flipped" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  random.seed(22) # 21
  random.shuffle(read_img_paths)
  assert constants.n_tr_tt == len(read_img_paths)
  # 80%
  split_bar = constants.n_tr
  tr_paths, tt_paths = read_img_paths[:split_bar], read_img_paths[split_bar:]
  for p in tr_paths:
    parts = p.split(s)
    name = 'DUT-LF-tr-tt-x8='+parts[-1]
    print(save_dir+name)
    os.rename(p, save_dir+name)
  for p in tt_paths:
    parts = p.split(s)
    name = 'test='+parts[-1]
    print(save_dir+name)
    os.rename(p, save_dir+name)
  print('success! - tr_tt_split')
  t.lap()
  return save_dir

def process_images(image_paths):
  from_shape = constants.from_shape
  to_shape = constants.to_shape
  partial_avg = np.zeros(to_shape, dtype=np.float32)
  partial_sum_sq_diff = np.zeros(to_shape, dtype=np.float32)
  u = 0
  for path in image_paths:
    image = Image.open(path)
    data = np.asarray(image, dtype=np.float32)
    partial_avg += data
    print(u)
    u = u + 1
    # Standard deviation computation is deferred to the main process
  return partial_avg, len(image_paths)


def reduce_results(results):
  total_avg = np.zeros_like(results[0][0])
  total_n = 0
  for avg, n in results:
    total_avg += avg
    total_n += n
  total_avg /= total_n
  return total_avg, total_n


def calculate_tr_avg_std(normalize_data_type=None):
  if normalize_data_type == 'APP':
    read_dir = './Datasets/APP/MINI_BATCH_DATA_TEST/'
    save_path = './Datasets/APP/MEAN_STD/' + 'mean_std.npz'
  else:  # TRAIN
    read_dir = './Datasets/TRAIN-DATA/'
    save_path = './Datasets/mean_std/SMART/' + 'mean_std.npz'

  if os.path.exists(save_path):
    print('avg and std already calculated!')
    with np.load(save_path) as tr_ms:
      return tr_ms['avg'], tr_ms['std']

  img_format = "png"  # Assuming image format is png
  read_img_paths_tr = glob.glob(read_dir + "ComplexBackground*." + img_format, recursive=False)
  # num_processes = cpu_count()  # Get the number of CPUs available
  num_processes = 1  # Get the number of CPUs available

  # Divide the work into chunks and process them in parallel
  chunk_size = len(read_img_paths_tr) // num_processes
  chunks = [read_img_paths_tr[i:i + chunk_size] for i in range(0, len(read_img_paths_tr), chunk_size)]

  with Pool(processes=num_processes) as pool:
    partial_results = pool.map(process_images, chunks)

  # Reduce the partial results to get the total average
  total_avg, total_n = reduce_results(partial_results)

  # Now calculate the standard deviation
  std = np.zeros_like(total_avg)
  u=0
  for chunk in chunks:
    for path in chunk:
      print(u)
      u=u+1
      image = Image.open(path)
      data = np.asarray(image, dtype=np.float32)
      std += np.square(data - total_avg)
  std = np.sqrt(std / (total_n - 1))

  np.savez(save_path, avg=total_avg, std=std)
  print('success! - avg and std are saved')
  return total_avg, std




def calculate_tr_min_max(save_dir):
  save_path = save_dir+'min_max.npz'
  print(save_path)
  if os.path.exists(save_path):
    print('min and max already calculated!')
    with np.load(save_path) as tr_ms:
      min_a, max_a = tr_ms['min'], tr_ms['max']
    return min_a, max_a
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if af=="x32":
    read_dir = dataset_root+ dn+"-tr-tt" + s
  else:
    read_dir = dataset_root+ dn+"-tr-tt-"+af + s
  read_img_paths_tr = glob.glob(read_dir+"DUT-LF-tr-tt-x8=*."+img_format, recursive=False)
  print('calculating min ...')
  from_shape = constants.from_shape
  to_shape = constants.to_shape
  min_a = np.full(to_shape,255,dtype=np.uint8)
  max_a = np.full(to_shape,0,dtype=np.uint8)
  for p in read_img_paths_tr:
    image = Image.open(p)
    data = np.asarray(image)
    a = data.reshape(from_shape, order='F')
    a = np.swapaxes(a, 1, 2)
    a = a.reshape(to_shape)
    min_a = np.stack((min_a,a)).min(axis=0)
  min_a = min_a.astype('float32')
  print(min_a.shape)
  print(min_a)
  np.savez(save_path,min_a=min_a, max_a=max_a)
  print('success! - min')
  t.lap()
  print('calculating max ...')
  for p in read_img_paths_tr:
    image = Image.open(p)
    data = np.asarray(image)
    a = data.reshape(from_shape, order='F')
    a = np.swapaxes(a, 1, 2)
    a = a.reshape(to_shape)
    max_a = np.stack((max_a,a)).max(axis=0)
  max_a = max_a.astype('float32')
  print(max_a.shape)
  print(max_a)
  print('success! - max')
  t.lap()
  np.savez(save_path,min_a=min_a, max_a=max_a)
  print('success! - min and max are saved')
  return min_a, max_a



if __name__ == "__main__":
  # 扁平化(把所有图片放在同一个文件夹下面)
  flatten_save_dir = dataset_root+ dn+"-flatten" + s
  if not os.path.exists(flatten_save_dir):
    os.makedirs(flatten_save_dir)
    flatten_dataset(flatten_save_dir)
    t.lap()
  # rotate x 8 times
  rotate_flip_save_dir = dataset_root+ dn+"-rotated-flipped" + s
  if not os.path.exists(rotate_flip_save_dir):
    os.makedirs(rotate_flip_save_dir)
    rotate_flip(rotate_flip_save_dir)
    # rotate_flip(rotate_flip_save_dir)

  # # Split the dataset into training and testing sets
  # if not os.path.exists(tr_tt_save_dir):
  #   os.makedirs(tr_tt_save_dir)
  #   tr_tt_split(tr_tt_save_dir)
  # # exit()
  # if af == "x32":
  #   tr_stats_save_dir = dataset_root+ dn+"-tensor" + s
  # else:
  #   tr_stats_save_dir = dataset_root+ dn+"-tensor-"+ af + s
  # if not os.path.exists(tr_stats_save_dir):
  #   os.makedirs(tr_stats_save_dir)
  # # Calculate the mean and standard deviation of the training set.
  # calculate_tr_avg_std(tr_stats_save_dir)
  # # calculate_tr_min_max(tr_stats_save_dir)
  # # if af == "x32":
  # #   tt_batch_save_dir = dataset_root+ dn+"-tensor" + s
  # # else:
  # #   tt_batch_save_dir = dataset_root+ dn+"-tensor-"+ af + s
  # # if not os.path.exists(tt_batch_save_dir):
  # #   os.makedirs(tt_batch_save_dir)
  # # utils.generate_test_batches(1,model_struct='LFACon',save_dir=tt_batch_save_dir, normAL=True,verbose=True)





# def to_sais(save_dir):
#   read_dir = dataset_root+ dn+"-tr-tt" + s
#   read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
#   for p in read_img_paths:
#     name = p.split(s)[-1][:-4]
#     print('processing', name, '...')
#     image = Image.open(p)
#     data = np.asarray(image)
#     from_shape = (9, 512, 9, 512, 3)
#     a = data.reshape(from_shape, order='F')
#     a = np.swapaxes(a, 1, 2)
#     for i in range(from_shape[0]):
#       for j in range(from_shape[2]):
#         img = Image.fromarray(a[i][j][:][:][:])
#         d = save_dir+name+s
#         if not os.path.exists(d):
#           os.makedirs(d)
#         img.save(d+str(i)+'_'+str(j)+'.bmp',img_format)
#   print('success! - to_sais')
#   t.lap()
#   return save_dir