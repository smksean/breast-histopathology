### Breast Histopathology Project Structure

### 1. __init__.py (inside app/)
Think of this file as a helper toolkit for image preparation. Before we give an image to the model, we must turn it into the right shape and scale.
Imports:

PIL.Image → opens and reads image files.
torchvision.transforms → resize, normalize, and convert to tensors.
torch → PyTorch's core library for tensors and models.

Key Functions:

get_transform(input_size):

Resizes the image to a fixed square (default 224 × 224).
Converts it to a tensor (numbers the model can read).
Normalizes the pixel values so they match the model's training settings (ImageNet mean/std).


load_image_as_tensor(path, input_size):

Opens an image from the given path.
Applies the transformation above.
Adds a batch dimension (unsqueeze(0)) so the model sees it as [1, C, H, W].



💡 Analogy: This is the kitchen prep station — slicing, washing, and plating the image so the "chef" (model) can work on it.

### 2. inference.py
This is the main engine that runs the trained model on new data.
BreastHistoModel class:
__init__:

Takes the path to your trained model file and the list of class names.
Chooses GPU if available, otherwise CPU.
Loads the model into memory.
Prepares the same transforms as training.

predict(img_tensor, return_label=False):

Runs the model on one prepared image tensor.
Uses softmax to get probabilities.
Finds the most likely class.
Optionally returns the label (benign/malignant).

predict_folder(folder_path, return_label=False):

Goes through all images in a folder.
Runs predictions for each image.
Averages all the probabilities to get a single final prediction.

💡 Analogy: This is the actual "chef" — given clean ingredients (preprocessed images), it makes the decision (benign or malignant).
### 3. utils.py
This is basically a copy of the preprocessing code from __init__.py. It's there so other scripts can import and use it without touching __init__.py.
💡 Analogy: Another prep station — same tools, just available for other kitchens.
### 4. inspect_checkpoint.py
Quick model inspection tool.
You run:
bashpython scripts/inspect_checkpoint.py models/my_model.pt
It loads the model and prints its architecture (layers, shapes).
💡 Analogy: Peeking inside the chef's brain to see how it's wired.

### 5. test_inference.py
Testing script for running the model outside the app.
Key Functions:

Loads the trained model file (.pt).
Prepares the same transforms.
predict_image: runs prediction on one image.
predict_folder: runs predictions on all images in a folder, then:

Uses majority vote for the final decision.
Averages probabilities for reporting.



💡 Analogy: A taste-testing table — you bring either one dish or a tray of dishes, and it decides which flavor dominates.
### 6. test_loader.py
More flexible command-line tool for testing.
Features:

Lets you specify:

Model path (--model)
Single image (--image)
Folder of images (--images_dir)
Aggregation method (--aggregate → mean, median, or vote).


Uses BreastHistoModel from inference.py.
Prints out probabilities and final prediction.

💡 Analogy: A more advanced taste tester — you can tell it how to combine multiple dishes' opinions.