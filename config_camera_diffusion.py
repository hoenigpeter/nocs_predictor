# General Training Settings
max_epochs = 100
batch_size = 16
test_batch_size = 1
shuffle_buffer = 1000
save_epoch_interval = 5
gpu_id = 0

iter_cnt = 100

#"cam_K": [909.9260796440973, 0.0, 643.5625, 0.0, 907.9168585002319, 349.017181282838, 0.0, 0.0, 1.0

#For the real data: [[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]]

#For the synthetic data: [[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]

fx = 591.0125
fy = 590.16775
cx = 322.525
cy = 244.11084

width = 640
height = 480

freeze_backbone = True

# Data Paths
train_data_root = "/ssd3/datasets_bop/camera_real275_datasets/camera_train/shard-{000000..000053}.tar"    # is different!
val_data_root = "/ssd3/datasets_bop/camera_real275_datasets/real275_val/scene_1.tar"
test_data_root = "/ssd3/datasets_bop/camera_real275_datasets/real275_test/scene_{2..6}.tar"
models_root = None

coco_json_path = "/ssd3/real_camera_dataset/real275_test.json"
test_images_root = "/ssd3/real_camera_dataset/real_test"

class_name = None
num_categories = 6
num_points_to_sample = 2000

# Directories for Saving Weights and Validation Images
weight_dir = "./weights_camera_diffusion_dino_bart"
val_img_dir = "./val_img_camera_diffusion_dino_bart"
test_img_dir = "./test_img_camera_diffusion_dino_bart"
ply_output_dir = "./plys_camera_diffusion_dino_bart"

with_transformer_loss = True
symmetry_type = 'instance_symmetries' # or 'category_symmetries'

noise_bound = 0.02  # 0.01
rotation_max_iterations = 2000
rotation_cost_threshold = 1e-12  # 1e-12

w_NOCS_bins = 0.0
w_NOCS_cont = 1.0
w_NOCS_ss = 0.0
w_seg = 0.0
w_Rot = 0.0
w_bg = 0.0

# Input Data Settings
image_size = 256

num_training_steps = 1000
num_inference_steps = 100

weight_file = 'generator_epoch_15.pth'

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
