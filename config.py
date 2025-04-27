# Dataset paths (update these to your actual paths!)
#train_dir = "/storage/prakhar/gravity_spy_net/Augmented_data/train"  # Directory with class folders
#val_dir = "/storage/prakhar/gravity_spy_net/Augmented_data/validation"      # Directory with class folders
test_dir = "./data"    # Directory with class folders

# Hyperparameters
batch_size = 256
epochs = 150

resize_x = 224  # Adjust based on model requirements
resize_y = 224
input_channels = 3
num_classes = 22  # Update based on GravitySpy classes

use_scheduler = True  # Use cosine annealing scheduler
learning_rate = 1e-3 #Intial LR
min_lr = 1e-7  # Minimum learning rate for scheduler