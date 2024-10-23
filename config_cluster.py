# General Training Settings
max_epochs = 100
batch_size = 64
test_batch_size = 1
shuffle_buffer = 1000
save_epoch_interval = 5
gpu_id = 0

iter_cnt = 100

freeze_backbone = True

# Data Paths
train_data_root = "/home/peter.hoenig/housecat6d_nocs_train/scene{01..34}.tar"
val_data_root = "/home/peter.hoenig/housecat6d_nocs_val/val_scene{1..2}.tar"
test_data_root = "/home/peter.hoenig/housecat6d_nocs_val/val_scene{1..2}.tar"
models_root = "/home/peter.hoenig/housecat6d/obj_models_ply_1000"

class_name = None

# Directories for Saving Weights and Validation Images
weight_dir = "./weights_vit"
val_img_dir = "./val_img_vit"
test_img_dir = "./test_img_vit"

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
augmentation = False
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
