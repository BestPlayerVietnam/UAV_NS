class Config:
    #IMG PARAMETERS
    IMG_TRAIN_COUNT = 200
    IMG_VALID_COUNT = 50
    IMG_HEIGHT = 720
    IMG_WIDTH = 1280
    PATCH_HEIGHT = 288
    PATCH_WIDTH = 288
    ORIG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
    PATCH_SIZE = (PATCH_WIDTH, PATCH_HEIGHT)
    NUM_CLASSES = 4
    CLASS_DISTRIBUTION =[
                        0.293, 0.304, 0.143, 0.26
                        ]
    TRAIN_PATCH_COUNT = 3000
    VALID_PATCH_COUNT = 750
    
    #PATH PARAMETERS
    ROOT_DIR = 'src_hd/dataset'
    TRAIN_SPLIT = 'train'
    VALID_SPLIT = 'valid'
    TEST_DIR = 'src_hd/test'
    SAVE_DIR = 'models'
    LOGS_DIR = 'logs'
    ANNOTATION_PATH = 'annotation.json'
    
    #LEARN PARAMETERS
    BATCH_SIZE = 4
    EPOCHS = 200
    LEARNING_RATE = 5e-3
    WEIGHT_DECAY = 1e-4
