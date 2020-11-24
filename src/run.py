from torch.utils.data import DataLoader
from dataset import Dataset
import segmentation_models_pytorch as smp
import glob
import model
import config
import augmentations

from catalyst.contrib.nn import BCEDiceLoss, RAdam, Lookahead, OneCycleLRWithWarmup
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import IouCallback, EarlyStoppingCallback, ClasswiseIouCallback


def main():

    train_image_list = sorted(glob.glob(pathname='../input/uavid-semantic-segmentation-dataset/train/train/*/Images/*.png', recursive=True))
    train_mask_list =  sorted(glob.glob(pathname='./trainlabels/*/TrainId/*.png', recursive=True))
    valid_image_list = sorted(glob.glob(pathname='../input/uavid-semantic-segmentation-dataset/valid/valid/*/Images/*.png', recursive=True))
    valid_mask_list =  sorted(glob.glob(pathname='./validlabels/*/TrainId/*.png', recursive=True))

    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

    train_dataset = Dataset(
        train_image_list, 
        train_mask_list, 
        augmentation=augmentations.get_training_augmentation(), 
        preprocessing=augmentations.get_preprocessing(preprocessing_fn),
        classes=config.CLASSES,
    )

    valid_dataset = Dataset(
        valid_image_list, 
        valid_mask_list, 
        augmentation=augmentations.get_validation_augmentation(), 
        preprocessing=augmentations.get_preprocessing(preprocessing_fn),
        classes=config.CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    base_optimizer = RAdam([
        {'params': model.MODEL.decoder.parameters(), 'lr': config.LEARNING_RATE}, 
        {'params': model.MODEL.encoder.parameters(), 'lr': 1e-4},
        {'params': model.MODEL.segmentation_head.parameters(), 'lr': config.LEARNING_RATE},
    ])
    optimizer = Lookahead(base_optimizer)
    criterion = BCEDiceLoss(activation=None)
    runner = SupervisedRunner()
    scheduler = OneCycleLRWithWarmup(
        optimizer, 
        num_steps=config.NUM_EPOCHS, 
        lr_range=(0.0016, 0.0000001),
        init_lr = config.LEARNING_RATE,
        warmup_steps=2
    )

    callbacks = [
        IouCallback(activation = 'none'),
        ClasswiseIouCallback(classes=config.CLASSES, activation = 'none'),
        EarlyStoppingCallback(patience=config.ES_PATIENCE, metric='iou', minimize=False),
        
    ]
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=config.LOGDIR,
        num_epochs=config.NUM_EPOCHS,
        # save our best checkpoint by IoU metric
        main_metric="iou",
        # IoU needs to be maximized.
        minimize_metric=False,
        # for FP16. It uses the variable from the very first cell
        fp16=config.FP16_PARAMS,
        # prints train logs
        verbose=True,
    )

if __name__ == '__main__':
    main()