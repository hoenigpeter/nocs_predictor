# General Training Settings
max_epochs = 100
batch_size = 16
test_batch_size = 1
shuffle_buffer = 1000
save_epoch_interval = 5
gpu_id = 0

iter_cnt = 100

freeze_backbone = True

# Data Paths
train_data_root = "/media/ssd2/peter/datasets/manibot/train_nocs/{000000..000008}.tar"
val_data_root = "/media/ssd2/peter/datasets/manibot/test_nocs/000009.tar"
test_data_root = "/media/ssd2/peter/datasets/manibot/test_nocs/000009.tar"
models_root = "/media/ssd2/peter/datasets/manibot/obj_models_ply_1000"

class_name = None
num_categories = 1

# Directories for Saving Weights and Validation Images
weight_dir = "./weights_manibot"
val_img_dir = "./val_img_manibot"
test_img_dir = "./test_img_manibot"

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
