####### General Training Settings
max_epochs = 100
batch_size = 2
test_batch_size = 1
shuffle_buffer = 1000
save_epoch_interval = 5
gpu_id = 0

###### INTRINSICS
fx = 591.0125
fy = 590.16775
cx = 322.525
cy = 244.11084

width = 640
height = 480

# Input Data Settings
image_size = 160    # dont change, else code goes brrrr (for now)

# Data Paths
train_data_root = "/ssd3/datasets_bop/camera/camera_pbr_all_dinos/shard-{000000..000509}.tar"    # is different!
val_data_root = "/ssd3/datasets_bop/camera_real275_datasets/real275_all_dinos/scene_1.tar"
test_data_root = val_data_root

coco_json_path = "/ssd3/real_camera_dataset/real275_test_3d_bbox.json"
test_images_root = "/ssd3/real_camera_dataset/real_test"

num_categories = 6

########## REFINEMENT - STUFF FOR ABLATIONS ###########
refinement = True           # use refinement? (you should)
num_refinement_steps = 6    # 6 works best

num_training_steps = 1000   # standard for DDPM (dont change)
num_inference_steps = 10    # standard for DPM++ solver (5 also good)
num_points_to_sample = 100  # number of points that are sampled from NOCS and depth cloud before Teaser++

#######################################################

########## AUGMENTATION
noisy_normals = True

################## EMBEDDINGS ###########################
with_dino_feat = False
with_bart_feat = False
with_cls_embedding = True          # embedding of cls id

with_dino_concat = False
dino_mode = "featup_pca"
# or: "dino_diff_nocs_pca"   -> original DiffusionNOCS dino + alread fit PCA
# or: "featup_pca"          -> Dinov2 # FeatUp 
# or: "dino_pca"            -> Dinov2 + fit_transform PCA
#########################################################

# Directories for Saving Weights and Validation Images
experiment_name = "camera_pbr_all_dinos_test"
weight_dir = "./weights_" + experiment_name
val_img_dir = "./val_img_" + experiment_name
test_img_dir = "./test_img_" + experiment_name

ply_output_dir = "./plys_" + experiment_name
pkl_output_dir = "./pkls_" + experiment_name
png_output_dir = "./pngs_" + experiment_name
bboxes_output_dir = "./bboxes_" + experiment_name

weight_file = 'generator_epoch_35.pth'

################# TEASER++ stuff dont touch this ##########
noise_bound = 0.02  # 0.01
rotation_max_iterations = 1000 # 1000
rotation_cost_threshold = 1e-12  # 1e-12
###########################################################

################## OPTIMIZER SETTINGS DONT TOUCH THIS #########
lr = 1e-4
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
warmup_steps = 1000
################## OPTIMIZER SETTINGS DONT TOUCH THIS #########

# Dataloader Settings
train_num_workers = 4
val_num_workers = 1

# Visualization Settings
iter_cnt = 1
num_imgs_log = 2
