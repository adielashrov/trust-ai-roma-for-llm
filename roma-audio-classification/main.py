# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import simpleaudio as sa
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import random
import datetime
import uuid
import math
import traceback
import warnings
import logging
import time
from scipy import stats
from scipy.stats import anderson

#This will print all types of warnings.
np.seterr(all='print')

# Configure logging and add suffix of current time to filename

# Get the current date and time

current_date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_file_name = f"app_{current_date_time}.log"
logging.basicConfig(filename=log_file_name, filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# i = the number in accents range: 1-60
# j = the number pronounced 0-9
# k = variation 0-49
# f'/kaggle/input/audio-mnist/data/0{i}/{j}_0{i}_{k}.wav'
output_dir = 'working'
input_dir = 'data/01'

device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables
BATCH_SIZE = 50
EPOCHS = 10
LR = 1e-3
NUM_OF_PERTRUBATIONS = 10000

num_feature_maps = 256
layers = [484, 128, 64, 32, 10]


feature_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((200,200)),
    T.ToTensor()
])

label_transform = T.Compose([
    T.ToTensor()
])

run_mode = 'robustness'  # 'train' 'robustness' or 'accuracy'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage
set_seed(0)  # Set the seed to an arbitrary value

def wav2melSpec(AUDIO_PATH):
    audio, sr = librosa.load(AUDIO_PATH)
    return librosa.feature.melspectrogram(y=audio, sr=sr)


def imgSpec(ms_feature):
    fig, ax = plt.subplots()
    ms_dB = librosa.power_to_db(ms_feature, ref=np.max)
    print(ms_feature.shape)
    img = librosa.display.specshow(ms_dB, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()  # This line ensures that the plot is displayed


def hear_audio(AUDIO_PATH):
    audio, sr = librosa.load(AUDIO_PATH, sr=None)  # Load the audio file
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)  # Normalize and convert to 16-bit
    # play_obj = sa.play_buffer(audio, 1, 2, sr)  # Play the audio
    # play_obj.wait_done()  # Wait until audio playback is done


def get_audio_info(path, show_melspec=False, label=None):
    spec = wav2melSpec(path)
    if label is not None:
        # print("Label:", label)
        pass
    if show_melspec is not False:
        imgSpec(spec)
    hear_audio(path)


def show_single_audio_file_spectogram():
    spec = wav2melSpec('data/01/0_01_0.wav')
    imgSpec(spec)
    audio_info = get_audio_info('data/01/0_01_0.wav', show_melspec=True)


# make two classes for dataset for features
# one returns mel spec
# one returns label
# split

class AudioDataset(Dataset):
    def __init__(self, path, feature_transform=None, label_transform=None, train=True, train_size=0.80):
        self.path = path
        self.file_list = []
        self.label_list = []
        self.feature_transform = feature_transform
        self.label_transform = label_transform
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                if filename[-3:] == "wav" and filename[0].isdigit():
                    self.file_list.append(os.path.join(dirname, filename))
                    self.label_list.append(int(filename[0]))
                # else:
                #     print("Skipping non valid filename:", filename)

        total_len = len(self.file_list)

        if train:
            self.file_list, self.label_list = self.file_list[:int(0.80 * total_len)], self.label_list[
                                                                                      :int(0.80 * total_len)]
        else:
            self.file_list, self.label_list = self.file_list[int(0.80 * total_len):], self.label_list[
                                                                                      int(0.80 * total_len):]

    def __getitem__(self, idx):
        try:
            spec = wav2melSpec(self.file_list[idx])
            spec = self.feature_transform(spec)
            label = self.label_list[idx]

            # Debugging: Check the range of values in the tensor
            assert spec.min() >= 0 and spec.max() <= 1, f"Tensor values out of range: min={spec.min()}, max={spec.max()}"

            return spec, label, self.file_list[idx]
        except:
            spec = wav2melSpec(self.file_list[0])
            spec = self.feature_transform(spec)
            label = self.label_list[idx]

            # Debugging: Check the range of values in the tensor
            assert spec.min() >= 0 and spec.max() <= 1, f"Tensor values out of range: min={spec.min()}, max={spec.max()}"

            return spec, label, self.file_list[idx]

    def __len__(self):
        return len(self.file_list)


def prepare_train_and_test_ds():
    train_ds = AudioDataset('data', feature_transform=feature_transform, label_transform=T.ToTensor(), train=True)
    test_ds = AudioDataset('data', feature_transform=feature_transform, label_transform=T.ToTensor(), train=False)

    logging.info(len(train_ds), len(test_ds))

    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    trainimages, trainlabels, traintext = next(iter(train_dataloader))

    logging.info(trainimages.shape)
    logging.info(trainlabels.shape)

    check = 10
    get_audio_info(traintext[check], True, trainlabels[check].item())


class AudioClassifier(nn.Module):
    def __init__(self, num_feature_maps, layers):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, num_feature_maps, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_feature_maps, num_feature_maps // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feature_maps // 2),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feature_maps // 2, num_feature_maps // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feature_maps // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_feature_maps // 4, 1, kernel_size=4, stride=1, padding=0),
            nn.Flatten()
        )

        self.classifier = nn.ModuleList(
            [nn.Linear(layers[i - 1], layers[i]) for i in range(1, len(layers))]
        )

    def forward(self, x):
        a = self.conv_layer(x)
        for layer in self.classifier:
            a = layer(a)
        return a  # as logits


def example_for_preparing_model_for_training():
    num_feature_maps = 256
    layers = [484, 128, 64, 32, 10]

    clf = AudioClassifier(num_feature_maps, layers).to(device)

    optim = torch.optim.Adam(lr=LR, params=clf.parameters(), betas=(0.5, 0.99))
    loss_fn = nn.CrossEntropyLoss()

def accuracy(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


def train_step(model, dataloader, optim, loss_fn, accuracy_fn):
    train_loss = 0.0
    train_acc = 0.0

    model.train()
    for batch, (X, y, txt) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_logits = model(X).to(device)
        y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1).to(device)

        acc = accuracy_fn(y_preds, y)
        loss = loss_fn(y_logits, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item()
        train_acc += acc

        if batch % 50 == 0:
            sample = random.randint(0, BATCH_SIZE - 2)
            logging.info(f"\tBatch {batch}: Train loss: {loss:.5f} | Train accuracy : {acc:.2f}%")
            # get_audio_info(txt[sample], label=y_preds[sample].item())
            logging.info("----------------------------------------")

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    logging.info(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def eval_step(model, dataloader, optim, loss_fn, accuracy_fn):
    test_loss = 0.0
    test_acc = 0.0

    model.eval()
    with torch.inference_mode():
        for batch, (X, y, txt) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_logits = model(X).to(device)
            y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1).to(device)

            acc = accuracy_fn(y_preds, y)
            loss = loss_fn(y_logits, y)

            test_loss += loss.item()
            test_acc += acc

            if batch % 50 == 0:
                sample = random.randint(0, BATCH_SIZE - 2)
                logging.info(f"\tBatch {batch}: Test loss: {loss:.5f} | Test accuracy : {acc:.2f}%")
                get_audio_info(txt[sample], label=y_preds[sample].item())
                logging.info("----------------------------------------")

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        logging.info(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")

    # Return the average test loss (to be used by the scheduler)
    return test_loss, test_acc

def save_model(model, path):
    current_date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_path = f"{path}/model_{current_date_time}.pth"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")


def train_model_function():
    train_ds = AudioDataset('data', feature_transform=feature_transform, label_transform=T.ToTensor(), train=True)
    test_ds = AudioDataset('data', feature_transform=feature_transform, label_transform=T.ToTensor(), train=False)

    logging.info(f"Train data set size: {len(train_ds)}, Test dataset size: {len(test_ds)}")

    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    trainimages, trainlabels, traintext = next(iter(train_dataloader))

    clf = AudioClassifier(num_feature_maps, layers).to(device)

    optim = torch.optim.Adam(clf.parameters(), lr=0.001)

    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim,
                                                    max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dataloader)),
                                                    epochs=EPOCHS,
                                                    anneal_strategy='linear')
    for epoch in range(EPOCHS):
        logging.info(f"Current Epoch: {epoch}")
        # Training step
        train_step(clf, train_dataloader, optim, loss_fn, accuracy)
        # Evaluation step - returns the validation loss
        val_loss, accuracy_score = eval_step(clf, test_dataloader, optim, loss_fn, accuracy)
        logging.info(f"Validation loss: {val_loss:.5f} | Validation accuracy: {accuracy_score:.2f}%")
        # Update the scheduler with validation loss
        scheduler.step(val_loss)
        if accuracy_score >= 90:
            break # model will be saved

    save_model(clf, 'models')

    logging.info('Done')


def add_noise_with_specified_norm(image, epsilon, norm_type='l2'):
    noise = torch.randn_like(image)
    if norm_type == 'l2':
        noise_norm = noise.norm(p=2)
    elif norm_type == 'l1':
        noise_norm = noise.norm(p=1)
    elif norm_type == 'linf':
        noise_norm = noise.norm(p=float('inf'))
    else:
        raise ValueError("Unsupported norm type")

    desired_norm = epsilon
    noise = noise * (desired_norm / noise_norm)

    perturbed_image = image + noise
    perturbed_image = perturbed_image.clamp(0, 1)

    return perturbed_image


def add_gaussian_noise(image, mean=0.0, std=0.1, epsilon=None):
    """
    Add Gaussian noise to an image tensor with specified mean and standard deviation.

    Parameters:
    - image (torch.Tensor): The original image tensor.
    - mean (float): The mean of the Gaussian noise.
    - std (float): The standard deviation of the Gaussian noise.
    - epsilon (float, optional): If specified, scales the noise to have an L2 norm of epsilon.

    Returns:
    - torch.Tensor: The image tensor with added noise.
    """
    noise = torch.normal(mean=mean, std=std, size=image.size())  # Generate noise from normal distribution

    if epsilon is not None:
        # Scale noise to have the desired L2 norm (epsilon)
        noise_norm = noise.norm(p=float('inf')) # use infinity norm
        noise = noise * (epsilon / noise_norm)

    # Add noise to the image
    perturbed_image = image + noise

    # Ensure pixel values are within [0, 1]
    perturbed_image = perturbed_image.clamp(0, 1) # should we use clamp or clip? TODO: what is the purpose of clamp?

    return perturbed_image


def add_noise_with_specified_norm(image, epsilon, norm_type='l2'):

    noise = torch.randn_like(image)
    if norm_type == 'l2':
        noise_norm = noise.norm(p=2)
    elif norm_type == 'l1':
        noise_norm = noise.norm(p=1)
    elif norm_type == 'linf':
        noise_norm = noise.norm(p=float('inf'))
    else:
        raise ValueError("Unsupported norm type")

    desired_norm = epsilon
    noise = noise * (desired_norm / noise_norm)

    perturbed_image = image + noise
    perturbed_image = perturbed_image.clamp(0, 1)

    # Verify the L2 norm of the difference
    difference = (perturbed_image - image).view(-1)
    l2_norm = difference.norm(p=2)

    # Adjust noise if the norm is larger than desired value
    while l2_norm > desired_norm:
        noise = noise * (desired_norm / l2_norm)
        perturbed_image = image + noise
        perturbed_image = perturbed_image.clamp(0, 1)
        difference = (perturbed_image - image).view(-1)
        l2_norm = difference.norm(p=2)

    return perturbed_image


def add_noise_roma_style(src_image, epsilon):
    src_image = src_image.float()

    # Generate uniform noise in [0, 1]
    noise = torch.rand_like(src_image)

    # Scale and shift noise to [-epsilon/2, epsilon/2]
    scaled_noise = epsilon * (noise - 0.5)

    # Add noise to image
    new_image = src_image + scaled_noise

    # Clamp to ensure values are in [0, 1]
    new_image = new_image.clamp(0, 1)

    # Debugging

    var1 = scaled_noise.abs().max().item()
    var2 = (new_image - src_image).abs().max().item()

    return new_image

# calculates the average predicted probabilities across all perturbed images
# and finds the index of the class with the highest average probability.
# This index, stored in `arg_max`,
# represents the most likely class for the perturbed images, excluding the true class.

def calculate_arg_max(stats):
    label_to = [[],[],[],[],[],[],[],[],[],[]]
    label_to_avg_value = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(stats)):
        index, value = stats[i]
        if not(label_to[index]):
            label_to[index] = []
        label_to[index].append(value)

    for i in range(len(label_to)):
        # logging.info(f"sum(label): {sum(label_to[i])} / len(label): {len(label_to[i])}")
        if len(label_to[i]) != 0:
            label_to_avg_value[i] = sum(label_to[i]) / len(label_to[i])

    max = 0
    arg_max = 0
    for i in range(len(label_to_avg_value)):
        if label_to_avg_value[i] and label_to_avg_value[i] > max:
            max = label_to_avg_value[i]
            arg_max = i

    return arg_max


# Define a custom warning handler that logs the call stack
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    logging.info(f"Warning: {message}")
    logging.info(f"Category: {category.__name__}")
    logging.info(f"File: {filename}, Line: {lineno}")
    logging.info("Call stack:")
    logging.info(''.join(traceback.format_stack()))


def check_model_robustness(num_of_samples=10):

    with warnings.catch_warnings(record=True) as w:
        statistics_at_work = []
        image_index = 0
        epsilon = 0.04
        threshold = 0.6
        clf = AudioClassifier(num_feature_maps, layers).to(device)
        clf.load_state_dict(torch.load('models/model_2024_07_27_22_00_49.pth'))
        clf.eval()

        # Load a single image from the dataset - same image every time
        test_ds = AudioDataset('data', feature_transform=feature_transform, label_transform=T.ToTensor(), train=False)
        if len(test_ds) == 0:
            raise ValueError("Dataset is empty. Please check the data loading logic.")

        test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)
        logging.info(len(test_dataloader))
        finale_table = np.zeros((len(test_dataloader), 11))
        for test_images, test_labels, test_text in test_dataloader:
            start_time = time.time()  # Start timing for this sample
            skip_image = False
            unique_id = uuid.uuid4()
            status = -1
            fitted_lambda = 0
            coxbox_anderson = None
            statistic = "nan"

            # create n perturbed images
            for i in range(NUM_OF_PERTRUBATIONS):
                # perturbed_image = add_gaussian_noise(test_images, std=0.1, epsilon=epsilon)
                perturbed_image = add_noise_roma_style(test_images, epsilon)
                # Verify the L infinty norm of the difference
                difference = (perturbed_image - test_images).view(-1)
                linf_norm = difference.abs().max()

                if linf_norm > epsilon:
                    logging.info(f"L-infinity norm of the perturbation: {linf_norm.item()}")
                    logging.info(f"Is norm <= epsilon? {linf_norm <= epsilon}")
                    logging.info("WARNING: Perturbation is larger than epsilon!")
                    # Additional debugging
                    max_diff_index = difference.abs().argmax()
                    logging.info(f"Max difference at index: {max_diff_index}")
                    logging.info(f"Original value: {test_images.view(-1)[max_diff_index].item()}")
                    logging.info(f"Perturbed value: {perturbed_image.view(-1)[max_diff_index].item()}")

                # Get the model's prediction
                y_logits = clf(perturbed_image.to(device))
                softmax_values = torch.softmax(y_logits, dim=1)

                highest_value, highest_index = torch.max(softmax_values, dim=1)

                sorted_values, sorted_indices = torch.sort(softmax_values, dim=1, descending=True)
                second_highest_value = sorted_values[:, 1]
                second_highest_index = sorted_indices[:, 1]

                statistics_at_work.append([unique_id, test_labels.item(),
                                           highest_index.item(), highest_value.item(),
                                           second_highest_index.item(), second_highest_value.item() ])


            second_highest_array = list(map(lambda x: x[4:], filter(lambda x: x[0] == unique_id, statistics_at_work)))
            second_highest_value_array = list(map(lambda x:x[1], second_highest_array))
            second_highest_value_array = np.array(second_highest_value_array)

            arg_max = calculate_arg_max(second_highest_array)

            # Calculating anderson darling test
            if len(second_highest_value_array) > 1:
                pred_anderson = anderson(second_highest_value_array, dist='norm')
            else:
                logging.error("Error: second_highest_value_array must contain more than one element., continuing to next image")
                continue
            if pred_anderson.statistic < pred_anderson.critical_values[0]: # ask Nati, why 0 and not other values?
                    logging.info("The second highest value is normally distributed")
                    normalize_Z_score = (threshold - np.mean(second_highest_value_array)) / np.std(second_highest_value_array)
                    p_value = stats.norm.cdf(abs(normalize_Z_score))
                    statistic = pred_anderson.statistic
                    status = 1
            else: # if the second-highest value array is not normally distributed
                # logging.info(f"The second highest value is NOT straightforward normally distributed! for uuid:{unique_id}")
                # See if we can apply the box-cox transformation
                first_element = second_highest_value_array[0]
                if np.all(second_highest_value_array == first_element) or any(second_highest_value_array <= 0):
                    logging.info(f"Cannot apply the Box-Cox transformation for uuid:{unique_id}")
                    normalize_Z_score = "nan"
                    p_value = "nan"
                    statistic = "nan"
                    status = 10
                else:
                    # ("Going to try to apply the box-cox transformation")
                    try:
                        shaped_cox_box, fitted_lambda = stats.boxcox(second_highest_value_array)
                    except Warning as w:
                        logging.info(f"stats.boxcox - Encountered problem error for uuid:{unique_id}" )
                        logging.info(f"Error:{w}")
                        traceback.print_exc()

                    # Re-check anderson value
                    coxbox_anderson = anderson(shaped_cox_box, dist='norm')
                    if coxbox_anderson.statistic < coxbox_anderson.critical_values[0]:
                        # logging.info("After box-cox, the second highest value is normally distributed for uuid:", unique_id)
                        if fitted_lambda == 0:
                            try:
                                # logging.info(f"Threshold:{threshold}")
                                Z_score = math.log(threshold)
                                status = 4
                            except Warning as w:
                                logging.warning(f"math.log(threshold) - Encountered problem error for uuid:{unique_id} with threshold:{threshold}")
                                logging.warning(f"Warning: {w}")
                                traceback.print_exc()
                        else:
                            try:
                                Z_score = ((threshold ** fitted_lambda) - 1) / fitted_lambda
                                status = 2
                            except Warning as w:
                                logging.info(f"((threshold ** fitted_lambda) - 1) / fitted_lambda - encountered overflow error for uuid: {unique_id} with threshold:{threshold}")
                                logging.info(f"Warning: {w}")
                                traceback.print_exc()

                        normalize_Z_score = (Z_score - np.mean(shaped_cox_box)) / np.std(shaped_cox_box)
                        p_value = stats.norm.cdf(abs(normalize_Z_score))
                        statistic = coxbox_anderson.statistic
                    else:
                        # logging.info(f"After box-cox, the second highest value is NOT normally distributed for uuid:{unique_id}")
                        normalize_Z_score = "nan"
                        p_value = "nan"
                        statistic = "nan"
                        status = 3

            # Calculate runtime for this sample
            sample_runtime = time.time() - start_time

            # Store the results in the final table
            finale_table[image_index, 0] = arg_max
            finale_table[image_index, 1] = test_labels.item()
            finale_table[image_index, 2] = fitted_lambda
            finale_table[image_index, 3] = statistic
            finale_table[image_index, 4] = coxbox_anderson.critical_values[0] if coxbox_anderson else "nan"
            finale_table[image_index, 5] = coxbox_anderson.significance_level[0] if coxbox_anderson else "nan"
            finale_table[image_index, 6] = "nan"
            finale_table[image_index, 7] = status
            finale_table[image_index, 8] = p_value
            finale_table[image_index, 9] = normalize_Z_score
            finale_table[image_index, 10] = sample_runtime  # Add runtime to the statistics

            image_index = image_index + 1
            logging.info(f"Number of samples processed: {image_index}" )

            # disable - run on the entire dataset
            # if image_index >= num_of_samples:
            #    logging.info("Breaking from loop")
            #    break

    current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    np.savetxt(f"results/normality_stats_{current_time}.csv", finale_table, delimiter=",",
               header="favorite_index,test_set_label, lambda, statistic, box_cox_critical_val ,"
                      "box_cox_sig_level, nan, status ,P_value, Z, runtime")

    return statistics_at_work


def check_model_accuracy(model_name, batch_size=1):
    clf = AudioClassifier(num_feature_maps, layers).to(device)
    model_full_name = ''.join(['models/', model_name])
    clf.load_state_dict(torch.load(model_full_name))
    clf.eval()

    test_ds = AudioDataset('data', feature_transform=feature_transform, label_transform=T.ToTensor(), train=False)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    # Print test_dataloader size
    logging.info(f"Test dataloader size: {len(test_dataloader)*batch_size}")

    correct = 0
    total = 0

    for test_images, test_labels, test_text in test_dataloader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        y_logits = clf(test_images)
        softmax_values = torch.softmax(y_logits, dim=1)
        highest_values, highest_indices = torch.max(softmax_values, dim=1)

        correct += (highest_indices == test_labels).sum().item()
        total += test_labels.size(0)
        logging.info(f"Correct: {correct} / Total: {total}")
    accuracy = (correct / total) * 100
    logging.info(f"{model_name} Accuracy: {accuracy:.2f}%")


def check_specific_model_accuracy():
    check_model_accuracy('model_2024_09_10_23_06_35.pth',batch_size=100)


def check_cuda_availability():
    # Check for CUDA availability
    if torch.cuda.is_available():
        logging.info('CUDA is available. Running on GPU.')
        # Optional: Log additional GPU details
        logging.info(f'GPU Name: {torch.cuda.get_device_name(0)}')
        logging.info(f'CUDA Version: {torch.version.cuda}')
    else:
        logging.info('CUDA is not available. Running on CPU.')

if __name__ == '__main__':
    # Override the default warning handler
    warnings.showwarning = custom_warning_handler

    current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    filename = f'results/stats_{current_time}.csv'

    if run_mode == 'train':
        train_model_function()
    elif run_mode == 'robustness':

        logging.info("Starting mnist audio model robustness testing...")

        check_cuda_availability()

        start_time_check_robustness = time.time()
        robustness_stats = check_model_robustness(10)

        # write the results to a csv file, one csv file per label
        df = pd.DataFrame(robustness_stats, columns=['UUID', 'TrueLabel', 'MaxIndex', 'MaxValue', 'SecMaxIndex', 'SecMaxValue'])
        df.to_csv(filename, index=False)
        elapsed_time = time.time() - start_time_check_robustness
        logging.info(f'Total runtime of check_model_robustness: {elapsed_time:.3f} seconds')
        logging.info(f'DataFrame saved to {filename}')
    elif run_mode == 'accuracy':
        check_specific_model_accuracy()
    else:
        logging.error("Invalid run mode. Please set the run_mode variable to 'train', 'robustness', or 'accuracy'.")