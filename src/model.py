import segmentation_models_pytorch as smp
import config


    # create segmentation model with pretrained encoder
def get_model():
    model = smp.FPN(
    encoder_name=config.ENCODER, 
    encoder_weights=config.ENCODER_WEIGHTS, 
    classes=len(config.CLASSES), 
    activation=config.ACTIVATION,
)
    return model
