import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from contrastyou import DATA_PATH
from contrastyou.dataloader.acdc_dataset import ACDCDataset
from semi_seg.augment import ACDCStrongTransforms

tra_transforms = ACDCStrongTransforms.pretrain
val_transforms = ACDCStrongTransforms.trainval

tra_dataset = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=tra_transforms, verbose=True)
val_dataset = ACDCDataset(root_dir=DATA_PATH, mode="val", transforms=val_transforms, verbose=True)

# Create DataLoader  
# since you are doing full supervision, you can simply do the following.

train_loader = DataLoader(tra_dataset, num_workers=4, batch_size=6, shuffle=True)
val_loader = DataLoader(val_dataset, num_workers=1, batch_size=6, shuffle=False)
for tra_data in train_loader:
    ((tra_img1, tra_gt_1), _), tra_filename, _, _ = tra_data

    # just show the first image on the batch
    plt.clf()
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(tra_img1[0].squeeze(), cmap="gray")
    plt.contour(tra_gt_1[0].squeeze())
    plt.subplot(222)

    plt.imshow(tra_img2[0].squeeze(), cmap="gray")
    plt.contour(tra_gt_2[0].squeeze())
    plt.show(block=False)
    plt.pause(1)
