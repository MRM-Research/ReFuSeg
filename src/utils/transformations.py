import albumentations as albu

def get_training_augmentation():
     
    train_transform = [
        albu.Resize(240, 240, always_apply=True),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0, rotate_limit=20, shift_limit=0.1, p=0.5, border_mode=0),
        albu.RandomCrop(height=224, width=224),
        albu.PadIfNeeded(min_height=240, min_width=240, always_apply=True, border_mode=0),
    ]
#         albu.ColorJitter(p=0.5),
#         albu.IAAAdditiveGaussianNoise(p=0.2),
#         albu.IAAPerspective(p=0.5),
    # albu.PadIfNeeded(min_height=240, min_width=240, always_apply=True, border_mode=0),
#         albu.OneOf(
#             [
#                 albu.CLAHE(p=1),
#                 albu.RandomBrightness(p=1),
#                 albu.RandomGamma(p=1),
#             ],
#             p=0.9,
#         ),

#         albu.OneOf(
#             [
#                 albu.IAASharpen(p=1),
#                 albu.Blur(blur_limit=3, p=1),
#                 albu.MotionBlur(blur_limit=3, p=1),
#             ],
#             p=0.9,
#         ),

#         albu.OneOf(
#             [
#                 albu.RandomContrast(p=1),
#                 albu.HueSaturationValue(p=1),
#             ],
#             p=0.9,
#         ),
    
    additional_targets = {
        #'image': 'image',
        'image1': 'image',
        'image2': 'image',
        'image3': 'image',
        #'seg': 'mask'
    }
    
    return albu.Compose(train_transform, additional_targets=additional_targets,is_check_shapes=False)
    #have to define 3 additional targets, 2 are already there, mask is used in documemntation to specify targets which dont have more complex changes like colorjitter to be done.

def get_validation_augmentation():
    test_transform = [
        albu.Resize(240, 240, always_apply=True),
        albu.PadIfNeeded(240, 240),
    ]
    
    additional_targets = {
        #'image': 'image',
        'image1': 'image',
        'image2': 'image',
        'image3': 'image',
        #'seg': 'mask'
    }
    
    return albu.Compose(test_transform, additional_targets=additional_targets,is_check_shapes=False)


def to_tensor(x, **kwargs):
    # return x.transpose(2, 0, 1).astype('float32')
    return x.astype('float32')

def preprocessing_fn(x):
    x /= 255.
    # x -= 0.5
    # x *= 2.
    return x

def get_preprocessing():
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        # albu.Lambda(image=preprocessing_fn, mask=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    
    additional_targets = {
        #'image': 'image',
        'image1': 'image',
        'image2': 'image',
        'image3': 'image',
        #'seg': 'mask'
    }
    
    return albu.Compose(_transform, additional_targets=additional_targets,is_check_shapes=False)
    