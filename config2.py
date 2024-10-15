# General Training Settings
max_epochs = 100
batch_size = 16
shuffle_buffer = 1000
save_epoch_interval = 5
gpu_id = 0

freeze_backbone = False

# Data Paths
train_data_root = "/media/ssd2/peter/datasets/housecat6d_nocs_train_with_rotation/scene{01..34}.tar"
val_data_root = "/media/ssd2/peter/datasets/housecat6d_nocs_val_with_info/val_scene{1..2}.tar"
models_root = "/media/ssd2/peter/datasets/obj_models_ply_5000"

# Directories for Saving Weights and Validation Images
weight_dir = "./weights_with_rot"
val_img_dir = "./val_img_with_rot"

# Input Data Settings
size = 128
num_bins = 50

# Optimizer Settings
lr = 1e-4
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 1e-6
warmup_steps = 1000

# Dataloader Settings
train_num_workers = 2
val_num_workers = 2

# Augmentation Settings
augmentation = True

# Visualization Settings
num_imgs_log = 8
arrow_length = 0.3
arrow_width = 0.03
arrow_colors = {
    'x': 'red',
    'y': 'green',
    'z': 'blue'
}
