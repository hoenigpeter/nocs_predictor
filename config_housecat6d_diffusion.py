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
test_data_root = "/ssd3/datasets_bop/housecat6d_nocs_val/test_scene{1..5}.tar"
models_root = "/ssd3/datasets_bop/housecat6d/obj_models_ply_1000"

coco_json_path = "/ssd3/datasets_bop/housecat6d_coco_test_1/coco_annotations.json"
test_images_root = "/ssd3/datasets_bop/housecat6d_coco_test_1"

class_name = None
num_categories = 10
num_points_to_sample = 2000

# Directories for Saving Weights and Validation Images
weight_dir = "./weights_housecat6d_diffusion_dino_bart"
val_img_dir = "./val_img_housecat6d_diffusion_dino_bart"
test_img_dir = "./test_img_housecat6d_diffusion_dino_bart"

with_transformer_loss = True
symmetry_type = 'instance_symmetries' # or 'category_symmetries'

noise_bound = 0.05

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
