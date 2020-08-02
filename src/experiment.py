from catalyst.dl import ConfigExperiment
import albumentations as albu
from torch.utils.data import DataLoader
from dataset import Dataset
import glob
import segmentation_models_pytorch as smp
import model
import config

train_image_list = sorted(glob.glob(pathname='../input/uavid-semantic-segmentation-dataset/train/train/*/Images/*.png', recursive=True))
train_mask_list =  sorted(glob.glob(pathname='./trainlabels/*/TrainId/*.png', recursive=True))
valid_image_list = sorted(glob.glob(pathname='../input/uavid-semantic-segmentation-dataset/valid/valid/*/Images/*.png', recursive=True))
valid_mask_list =  sorted(glob.glob(pathname='./validlabels/*/TrainId/*.png', recursive=True))

preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

class Experiment(ConfigExperiment):

    @staticmethod
    def get_training_augmentation():
        train_transform = [

            albu.Resize(576, 1024, p=1),
            albu.HorizontalFlip(p=0.5),

            albu.OneOf([
                albu.RandomBrightnessContrast(
                  brightness_limit=0.4, contrast_limit=0.4, p=1),
                albu.CLAHE(p=1),
                albu.HueSaturationValue(p=1)
                ],
                p=0.9,
            ),

            albu.IAAAdditiveGaussianNoise(p=0.2),
        ]
        return albu.Compose(train_transform)

    @staticmethod
    def get_validation_augmentation():
        test_transform = [albu.Resize(576, 1024, p=1),
        ]
        return albu.Compose(test_transform)

    @staticmethod
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing(self, preprocessing_fn):
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]
        return albu.Compose(_transform)

    def get_datasets(self, **kwargs):
        train_dataset = Dataset(
            train_image_list, 
            train_mask_list, 
            augmentation=self.get_training_augmentation(), 
            preprocessing=self.get_preprocessing(preprocessing_fn),
            classes=config.CLASSES,
        )

        valid_dataset = Dataset(
            valid_image_list, 
            valid_mask_list, 
            augmentation=self.get_validation_augmentation(), 
            preprocessing=self.get_preprocessing(preprocessing_fn),
            classes=config.CLASSES,
        )

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

        loaders = {
         "train": train_loader,
            "valid": valid_loader
        }
        return loaders

