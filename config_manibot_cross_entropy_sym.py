# General Training Settings
max_epochs = 100
batch_size = 16
test_batch_size = 1
shuffle_buffer = 1000
save_epoch_interval = 5
gpu_id = 0

iter_cnt = 100

fx = 605.463134765625
fy = 604.449951171875
cx = 322.2218322753906
cy = 236.83627319335938

width = 640
height = 480

freeze_backbone = True

# Data Paths
train_data_root = "/ssd3/datasets_bop/manibot/train_nocs_3/{000000..000008}.tar"
val_data_root = "/ssd3/datasets_bop/manibot/test_nocs/000009.tar"
test_data_root = "/ssd3/datasets_bop/manibot/test_nocs/000009.tar"
models_root = "/ssd3/datasets_bop/manibot/obj_models_ply_1000"

coco_json_path = "/ssd3/datasets_bop/manibot/test_images_real/coco_annotations.json"
test_images_root = "/ssd3/datasets_bop/manibot/test_images_real"

class_name = None
num_categories = 1

# Directories for Saving Weights and Validation Images
weight_dir = "./weights_manibot_cross_entropy_sym"
val_img_dir = "./val_img_manibot_cross_entropy_sym"
test_img_dir = "./test_img_manibot_cross_entropy_sym"

with_transformer_loss = True
symmetry_type = 'category_symmetries' # or 'category_symmetries'

w_NOCS_bins = 0.0
w_NOCS_cont = 1.0
w_NOCS_ss = 0.0
w_seg = 0.0
w_Rot = 0.0
w_bg = 0.0

# Input Data Settings
size = 128
num_bins = 256

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
