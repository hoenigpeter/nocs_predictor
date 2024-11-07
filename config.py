# General Training Settings
max_epochs = 100
batch_size = 16
test_batch_size = 1
shuffle_buffer = 1000
save_epoch_interval = 5
gpu_id = 0

iter_cnt = 100

fx = 705.4699707031250000
fy = 703.0218505859375000
cx = 554.1227923106198432
cy = 432.4074733765810379 

width = 1096
height = 852

freeze_backbone = True

# Data Paths
train_data_root = "/ssd3/datasets_bop/housecat6d_nocs_train/scene{01..34}.tar"
val_data_root = "/ssd3/datasets_bop/housecat6d_nocs_val/val_scene{1..2}.tar"
#test_data_root = "/ssd3/datasets_bop/housecat6d_nocs_test/test_scene{1..5}.tar"
test_data_root = "/ssd3/datasets_bop/housecat6d_nocs_test_new/test_scene1.tar"
models_root = "/ssd3/datasets_bop/housecat6d/obj_models_ply_5000"

class_name = 2

# Directories for Saving Weights and Validation Images
weight_dir = "./weights_2"
val_img_dir = "./val_img_2"
test_img_dir = "./test_img_2"

w_NOCS_bins = 1.0
w_NOCS_cont = 1.0
w_NOCS_ss = 1.0
w_seg = 1.0
w_Rot = 1.0
w_bg = 0

# Input Data Settings
size = 128
num_bins = 50

# Optimizer Settings
lr = 1e-5
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
warmup_steps = 1000

# Dataloader Settings
train_num_workers = 2
val_num_workers = 2

# Augmentation Settings
augmentation = True
center_crop = False

# Visualization Settings
num_imgs_log = 8
arrow_length = 0.3
arrow_width = 0.03
arrow_colors = {
    'x': 'red',
    'y': 'green',
    'z': 'blue'
}
