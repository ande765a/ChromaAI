from sklearn.model_selection import train_test_split
import os

data_path = "all_data"
for category in os.listdir(data_path):
    category_path = os.path.join(data_path, category)
    images = os.listdir(category_path) 
    train_images, validation_images = train_test_split(images, test_size=0.2, random_state=42)

    training_path = os.path.join(category_path, "training")
    if not os.path.exists(training_path): 
        os.makedirs(training_path)
    validation_path = os.path.join(category_path, "validation")
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    for train_image in train_images:
        os.rename(os.path.join(category_path, train_image), os.path.join(training_path, train_image))
    for validation_image in validation_images:
        os.rename(os.path.join(category_path, validation_image), os.path.join(validation_path, validation_image))