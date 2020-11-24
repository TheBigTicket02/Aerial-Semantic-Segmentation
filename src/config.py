ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['clutter', 'building', 'road', 'static_car', 'tree', 'vegetation', 'human', 'moving_car']
ACTIVATION = 'sigmoid'
BATCH_SIZE = 6
FP16_PARAMS = dict(opt_level="01")
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
ES_PATIENCE = 7
LOGDIR = "./logs"