# General Training Settings
max_epochs = 100
batch_size = 1
test_batch_size = 1
shuffle_buffer = 1000
save_epoch_interval = 5
gpu_id = 0

iter_cnt = 100

#"cam_K": [909.9260796440973, 0.0, 643.5625, 0.0, 907.9168585002319, 349.017181282838, 0.0, 0.0, 1.0

fx = 909.9260796440973
fy = 907.9168585002319
cx = 643.5625
cy = 349.017181282838

width = 1280
height = 720

# fx = 605.463134765625
# fy = 604.449951171875
# cx = 322.2218322753906
# cy = 236.83627319335938

# width = 640
# height = 480

freeze_backbone = True

# Data Paths
train_data_root = "/ssd3/datasets_bop/manibot/train_nocs_3/{000000..000008}.tar"
val_data_root = "/ssd3/datasets_bop/manibot/test_nocs/000009.tar"
test_data_root = "/ssd3/datasets_bop/manibot/test_nocs/000009.tar"
models_root = "/ssd3/datasets_bop/manibot/obj_models_ply_1000"

coco_json_path = "/ssd3/datasets_bop/manibot/test_images_zed_cart_1/coco_annotations.json"
test_images_root = "/ssd3/datasets_bop/manibot/test_images_zed_cart_1"

class_name = None
num_categories = 1

# Directories for Saving Weights and Validation Images
weight_dir = "./weights_manibot_diffusion_dino"
val_img_dir = "./val_img_manibot_diffusion_dino"
test_img_dir = "./test_img_manibot_diffusion_dino"

with_transformer_loss = True
symmetry_type = 'instance_symmetries' # or 'category_symmetries'

noise_bound = 0.01

w_NOCS_bins = 0.0
w_NOCS_cont = 1.0
w_NOCS_ss = 0.0
w_seg = 0.0
w_Rot = 0.0
w_bg = 0.0

# Input Data Settings
size = 128
num_bins = 50

# Optimizer Settings
lr = 1e-4
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
warmup_steps = 1000

# Dataloader Settings
train_num_workers = 2
val_num_workers = 1

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
