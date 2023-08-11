import os

import geopandas as gpd
import imageio
import matplotlib.pyplot as plt
import mercantile
import numpy as np
import pandas as pd
import rasterio as rasterio
import torch
from rasterio.rio.rasterize import rasterize
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm

from models.semanticsegmentation.FracTAL_ResUNet import FracTAL_ResUNet_cmtsk

df = ...
image_folder = ...

gdf = gpd.GeoDataFrame(df, geometry="geometry")

# ============================ #
# user-specified hyperparameters
# ============================ #
epochs = 100
lr = 0.001
lr_decay = None
n_filters = 16
batch_size = 8
n_classes = 1
model_type = 'resunet-d6'
codes_to_keep = [1]
ctx_name = 'gpu'
gpu_id = 0
boundary_kernel_size = (2, 2)
trained_model = None  # train from scratch

train_names = []
val_names = []
test_names = []
train_names_label = []
val_names_label = []
test_names_label = []


def get_tiles_for_geometry(geometry, zoom):
    bounds = geometry.bounds
    min_lon, min_lat, max_lon, max_lat = bounds
    tiles = mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=[zoom])
    return list(tiles)


zoom_level = 17
gdf['tiles'] = gdf['geometry'].apply(lambda geom: get_tiles_for_geometry(geom, zoom_level))

for _, row in gdf.iterrows():
    for tile in row['tiles']:
        image_path = os.path.join(image_folder, str(tile.z), str(tile.x), f"{tile.y}.jpg")
        lon_min, lat_min, lon_max, lat_max = mercantile.bounds(tile.x, tile.y, tile.z)
        src_image = imageio.imread(image_path)
        raster_labels = rasterize([row['geometry']], out_shape=src_image.shape[:2],
                                  transform=rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max,
                                                                           src_image.shape[1], src_image.shape[0]))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].imshow(src_image)
ax[0].set_title('Matplot image')

extent = raster_labels
ax[1].imshow(extent, cmap='Greys_r')
ax[1].set_title('Extent labels')

plt.show()


def train_model(train_dataloader, model, optimizer, epoch, args):
    # Initialize metrics
    cumulative_loss = 0
    accuracy = 0
    f1_score = 0
    mcc = 0
    dice_score = 0

    device = torch.device('cpu') if args['ctx_name'] == 'cpu' else torch.device('cuda', args['gpu'])
    model = model.to(device)

    model.train()
    for batch_i, (img, extent, boundary, distance, hsv, mask) in enumerate(
            tqdm(train_dataloader, desc='Training epoch {}'.format(epoch))):
        img = img.to(device)
        extent = extent.to(device)
        boundary = boundary.to(device)
        distance = distance.to(device)
        hsv = hsv.to(device)
        mask = mask.to(device)
        nonmask = torch.ones(extent.shape).to(device)

        logits, bound, dist, convc = model(img)

        # Multi-task loss
        # Not sure what to do with the masks here.
        jaccard = BinaryJaccardIndex()
        loss_extent = torch.sum(1 - jaccard(logits, extent))
        loss_boundary = torch.sum(1 - jaccard(bound, boundary))
        loss_distance = torch.sum(1 - jaccard(dist, distance))
        loss_hsv = torch.sum(1 - jaccard(convc, hsv))

        loss = 0.25 * (loss_extent + loss_boundary + loss_distance + loss_hsv)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item()

        # Update metrics based on every batch
        logits_reshaped = logits.view(logits.size(0), -1)
        extent_reshaped = extent.view(extent.size(0), -1)
        mask_reshaped = mask.view(mask.size(0), -1)
        nonmask_idx = torch.nonzero(mask_reshaped)
        logits_masked = torch.gather(logits_reshaped, 1, nonmask_idx)
        extent_masked = torch.gather(extent_reshaped, 1, nonmask_idx)

        # Accuracy
        extent_predicted_classes = torch.ceil(logits_masked - 0.5)
        accuracy += torch.sum(extent_masked == extent_predicted_classes).item()

        # F1 Score
        probabilities = torch.stack((1 - logits_masked, logits_masked), dim=1)
        _, predicted_classes = torch.max(probabilities, 1)
        f1_score += torch.sum(extent_masked == predicted_classes).item()

        # TODO: MCC metric
        # TODO: Dice score

    num_batches = len(train_dataloader)
    accuracy /= num_batches * args['batch_size']
    f1_score /= num_batches * args['batch_size']
    mcc /= num_batches * args['batch_size']
    dice_score /= num_batches * args['batch_size']

    return cumulative_loss, accuracy, f1_score, mcc, dice_score


def evaluate_model(val_dataloader, model, epoch, args):
    # Initialize metrics
    cumulative_loss = 0
    accuracy = 0
    f1_score = 0
    mcc = 0
    dice_score = 0

    device = torch.device('cpu') if args['ctx_name'] == 'cpu' else torch.device('cuda', args['gpu'])
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for batch_i, (img, extent, boundary, distance, hsv, mask) in enumerate(
                tqdm(val_dataloader, desc='Validation epoch {}'.format(epoch))):
            img = img.to(device)
            extent = extent.to(device)
            boundary = boundary.to(device)
            distance = distance.to(device)
            hsv = hsv.to(device)
            mask = mask.to(device)
            nonmask = torch.ones(extent.shape).to(device)

            logits, bound, dist, convc = model(img)

            # Multi-task loss
            # Not sure what to do with the masks here.
            jaccard = BinaryJaccardIndex()
            loss_extent = torch.sum(1 - jaccard(logits, extent))
            loss_boundary = torch.sum(1 - jaccard(bound, boundary))
            loss_distance = torch.sum(1 - jaccard(dist, distance))
            loss_hsv = torch.sum(1 - jaccard(convc, hsv))

            loss = 0.25 * (loss_extent + loss_boundary + loss_distance + loss_hsv)

            cumulative_loss += loss.item()

            # Update metrics based on every batch
            logits_reshaped = logits.view(logits.size(0), -1)
            extent_reshaped = extent.view(extent.size(0), -1)
            mask_reshaped = mask.view(mask.size(0), -1)
            nonmask_idx = torch.nonzero(mask_reshaped)
            logits_masked = torch.gather(logits_reshaped, 1, nonmask_idx)
            extent_masked = torch.gather(extent_reshaped, 1, nonmask_idx)

            # Accuracy
            extent_predicted_classes = torch.ceil(logits_masked - 0.5)
            accuracy += torch.sum(extent_masked == extent_predicted_classes).item()

            # F1 Score
            probabilities = torch.stack((1 - logits_masked, logits_masked), dim=1)
            _, predicted_classes = torch.max(probabilities, 1)
            f1_score += torch.sum(extent_masked == predicted_classes).item()

            # TODO: MCC metric
            # TODO: Dice score

    num_batches = len(val_dataloader)
    accuracy /= num_batches * args['batch_size']
    f1_score /= num_batches * args['batch_size']
    mcc /= num_batches * args['batch_size']
    dice_score /= num_batches * args['batch_size']

    return cumulative_loss, accuracy, f1_score, mcc, dice_score


def run_africa(train_names, val_names, test_names, train_names_label, val_names_label, test_names_label,
               trained_model=None,
               epochs=100, lr=0.001, lr_decay=None, n_filters=16, batch_size=8,
               n_classes=1, codes_to_keep=[1, 2],
               boundary_kernel_size=3,
               ctx_name='cpu',
               gpu_id=0
               ):
    # Set PyTorch device
    device = torch.device('cpu') if ctx_name == 'cpu' else torch.device('cuda', gpu_id)

    folder_name = f"{model_type}_nfilter-{n_filters}_bs-{batch_size}_lr-{lr}"
    if lr_decay:
        folder_name = f"{folder_name}_lrdecay-{lr_decay}"

    model = FracTAL_ResUNet_cmtsk(nfilters_init=n_filters, depth=depth, NClasses=n_classes)
    model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = os.path.join('../experiments/', folder_name)
    # if not os.path.isdir(save_path):
    #    os.makedirs(save_path)
    save_model_name = os.path.join(save_path, "model.pth")

    # Arguments
    args = {}
    args['batch_size'] = batch_size
    args['ctx_name'] = ctx_name
    args['gpu'] = gpu_id

    # Define train/val/test splits
    # Not sure how to convert AirbusMasked
    train_dataset = AirbusMasked(
        fold='train',
        image_names=train_names,
        label_names=train_names_label,
        classes=codes_to_keep,
        boundary_kernel_size=boundary_kernel_size)
    val_dataset = AirbusMasked(
        fold='val',
        image_names=val_names,
        label_names=val_names_label,
        classes=codes_to_keep,
        boundary_kernel_size=boundary_kernel_size)
    test_dataset = AirbusMasked(
        fold='test',
        image_names=test_names,
        label_names=test_names_label,
        classes=codes_to_keep,
        boundary_kernel_size=boundary_kernel_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    if lr_decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    else:
        scheduler = None

    # Containers for metrics to log
    train_metrics = {'train_loss': [], 'train_acc': [], 'train_f1': [],
                     'train_mcc': [], 'train_dice': []}
    val_metrics = {'val_loss': [], 'val_acc': [], 'val_f1': [],
                   'val_mcc': [], 'val_dice': []}
    best_mcc = 0.0

    # Training loop
    for epoch in range(1, epochs + 1):

        # Training set
        train_loss, train_accuracy, train_f1, train_mcc, train_dice = train_model(
            train_dataloader, model, optimizer, epoch, args)

        # Training set metrics
        train_loss_avg = train_loss / len(train_dataset)
        train_metrics['train_loss'].append(train_loss_avg)
        train_metrics['train_acc'].append(train_accuracy)
        train_metrics['train_f1'].append(train_f1)
        train_metrics['train_mcc'].append(train_mcc)
        train_metrics['train_dice'].append(train_dice)

        # Validation set
        val_loss, val_accuracy, val_f1, val_mcc, val_dice = evaluate_model(
            val_dataloader, model, epoch, args)

        # Validation set metrics
        val_loss_avg = val_loss / len(val_dataset)
        val_metrics['val_loss'].append(val_loss_avg)
        val_metrics['val_acc'].append(val_accuracy)
        val_metrics['val_f1'].append(val_f1)
        val_metrics['val_mcc'].append(val_mcc)
        val_metrics['val_dice'].append(val_dice)

        print("Epoch {}:".format(epoch))
        print("Train loss {:0.3f}, accuracy {:0.3f}, F1-score {:0.3f}, MCC: {:0.3f}, Dice: {:0.3f}".format(
            train_loss_avg, train_accuracy, train_f1, train_mcc, train_dice))
        print("Val loss {:0.3f}, accuracy {:0.3f}, F1-score {:0.3f}, MCC: {:0.3f}, Dice: {:0.3f}".format(
            val_loss_avg, val_accuracy, val_f1, val_mcc, val_dice))

        # Save model based on best MCC metric
        if val_mcc > best_mcc:
            torch.save(model.state_dict(), save_model_name)
            best_mcc = val_mcc

        # Save metrics
        metrics = pd.concat([pd.DataFrame(train_metrics), pd.DataFrame(val_metrics)], axis=1)
        metrics.to_csv(os.path.join(save_path, 'metrics.csv'), index=False)

        # Visualize
        plt.figure()
        plt.plot(np.arange(1, epoch + 1), train_metrics['train_loss'], label='Train loss')
        plt.plot(np.arange(1, epoch + 1), val_metrics['val_loss'], label='Val loss')
        plt.legend()
        plt.title('Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(save_path, 'Loss.png'))
        plt.close()

        plt.figure()
        plt.plot(np.arange(1, epoch + 1), train_metrics['train_mcc'], label='Train MCC')
        plt.plot(np.arange(1, epoch + 1), val_metrics['val_mcc'], label='Val MCC')
        plt.legend()
        plt.title('MCC')
        plt.xlabel('Epoch')
        plt.ylabel('MCC')
        plt.savefig(os.path.join(save_path, 'MCC.png'))
        plt.close()


run_africa(train_names, val_names, test_names,
           train_names_label, val_names_label, test_names_label,
           trained_model=trained_model,
           epochs=epochs, lr=lr, lr_decay=lr_decay, n_filters=n_filters, batch_size=batch_size,
           n_classes=n_classes, model_type=model_type,
           codes_to_keep=codes_to_keep,
           ctx_name=ctx_name,
           gpu_id=gpu_id,
           boundary_kernel_size=boundary_kernel_size)
