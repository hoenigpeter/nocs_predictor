import json
import matplotlib.pyplot as plt

def load_and_parse_json(file_path):
    training_iterations = []
    validation_epochs = []
    
    binary_nocs_losses = []
    regression_nocs_losses = []
    masked_nocs_losses = []
    seg_losses = []
    
    val_binary_nocs_losses = []
    val_regression_nocs_losses = []
    val_masked_nocs_losses = []
    val_seg_losses = []

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        if 'iteration' in entry:  # Training entry
            training_iterations.append(entry['iteration'])
            binary_nocs_losses.append(entry['binary_nocs_loss'])
            regression_nocs_losses.append(entry['regression_nocs_loss'])
            masked_nocs_losses.append(entry['masked_nocs_loss'])
            seg_losses.append(entry['seg_loss'])
        else:  # Validation entry
            validation_epochs.append(entry['epoch'])
            val_binary_nocs_losses.append(entry['val_binary_nocs_loss'])
            val_regression_nocs_losses.append(entry['val_regression_nocs_loss'])
            val_masked_nocs_losses.append(entry['val_masked_nocs_loss'])
            val_seg_losses.append(entry['val_seg_loss'])

    return (training_iterations, validation_epochs, binary_nocs_losses, 
            regression_nocs_losses, masked_nocs_losses, seg_losses,
            val_binary_nocs_losses, val_regression_nocs_losses, 
            val_masked_nocs_losses, val_seg_losses)

def plot_losses(training_iterations, validation_epochs, binary_nocs_losses, 
                regression_nocs_losses, masked_nocs_losses, seg_losses,
                val_binary_nocs_losses, val_regression_nocs_losses, 
                val_masked_nocs_losses, val_seg_losses):
    
    # Plot training losses
    plt.figure(figsize=(12, 8))
    
    # Binary NOCS Loss
    plt.subplot(2, 2, 1)
    plt.plot(training_iterations, binary_nocs_losses, label='Training Binary NOCS Loss')
    if val_binary_nocs_losses:
        plt.plot(validation_epochs, val_binary_nocs_losses, 'ro-', label='Validation Binary NOCS Loss')
    plt.title('Binary NOCS Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    # Regression NOCS Loss
    plt.subplot(2, 2, 2)
    plt.plot(training_iterations, regression_nocs_losses, label='Training Regression NOCS Loss')
    if val_regression_nocs_losses:
        plt.plot(validation_epochs, val_regression_nocs_losses, 'ro-', label='Validation Regression NOCS Loss')
    plt.title('Regression NOCS Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    # Masked NOCS Loss
    plt.subplot(2, 2, 3)
    plt.plot(training_iterations, masked_nocs_losses, label='Training Masked NOCS Loss')
    if val_masked_nocs_losses:
        plt.plot(validation_epochs, val_masked_nocs_losses, 'ro-', label='Validation Masked NOCS Loss')
    plt.title('Masked NOCS Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    # Segmentation Loss
    plt.subplot(2, 2, 4)
    plt.plot(training_iterations, seg_losses, label='Training Segmentation Loss')
    if val_seg_losses:
        plt.plot(validation_epochs, val_seg_losses, 'ro-', label='Validation Segmentation Loss')
    plt.title('Segmentation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
file_path = 'loss_log.json'
(training_iterations, validation_epochs, binary_nocs_losses, 
 regression_nocs_losses, masked_nocs_losses, seg_losses,
 val_binary_nocs_losses, val_regression_nocs_losses, 
 val_masked_nocs_losses, val_seg_losses) = load_and_parse_json(file_path)

plot_losses(training_iterations, validation_epochs, binary_nocs_losses, 
            regression_nocs_losses, masked_nocs_losses, seg_losses,
            val_binary_nocs_losses, val_regression_nocs_losses, 
            val_masked_nocs_losses, val_seg_losses)
