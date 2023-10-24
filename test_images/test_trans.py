from torchvision import transforms
from ..src.preprocess import VariableAspectRatio, Squeezing,GaussianNoise,Shading
from ..src.config import INPUT_SIZE, SQUEEZE_RATIO
from PIL import Image

TRANSFORMS_TRAIN = transforms.Compose(
    [
        transforms.Grayscale(),
        GaussianNoise(),
        transforms.GaussianBlur(3, sigma=(2.5, 3.5)),
        transforms.RandomPerspective(fill=255,distortion_scale=0.1),
        # VariableAspectRatio(INPUT_SIZE),
        # Squeezing(INPUT_SIZE, SQUEEZE_RATIO),
        # transforms.RandomCrop(INPUT_SIZE),
        transforms.ToTensor(),
        
    ]
)

img = Image.open('./test_images/syn/compare/CourierStd_284.png')
img_ = TRANSFORMS_TRAIN(img)
img_pil = transforms.ToPILImage()(img_)
img_pil.save('./test_images/trans/test.png')