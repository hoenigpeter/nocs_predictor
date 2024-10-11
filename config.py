max_epochs = 100
batch_size = 16
shuffle_buffer = 1000

train_data_root = "/ssd3/datasets_bop/housecat6d_nocs_train_with_rotation/scene{01..34}.tar"
val_data_root = "/ssd3/datasets_bop/housecat6d_nocs_val_with_info/val_scene{1..2}.tar"

weight_dir = "./weights"
val_img_dir = "./val_img"

size = 256
num_bins = 100

lr = 1e-4
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 1e-6

train_num_workers = 16
val_num_workers = 2

### VISUALIZATION
arrow_length = 0.3
arrow_width = 0.03
arrow_colors = {
    'x': 'red',
    'y': 'green',
    'z': 'blue'
}