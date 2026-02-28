from shurikodecontext.ml_model.m import Create_ResNet_Shurikode
from typing import Union, cast
from PIL.Image import Image
from torch import Tensor
from shurikodecontext.utils import find_device, ModelSizes, ModelType
import torchvision.transforms.v2 as transforms
import PIL.Image

import torch


class ShurikodeDecoder:
    __image_tensorizer = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    def __init__(self, size: ModelSizes = "M"):
        """
        Initializes a `shurikode_decoder` object.

        :param `m`: The model to be used in the decoder to decode the images. The possible models are 'r18', 'r34' and
        'r50'.
        """
        self.__device = find_device()

        model_type: ModelType = "r50"
        if size == "M":
            model_type = "r34"
        elif size == "S":
            model_type = "r18"

        self.__m = Create_ResNet_Shurikode(
            model_type, 257, self.__device, group_norm=False
        ).eval()

    def __call__(self, img: Union[Image, Tensor]) -> tuple[int, float]:
        """
        Given a single Pillow `Image` or a `torch.Tensor` representing just one image (both as a 3D tensor or as a 4D
        tensor of batch size 1), returns the shurikode label of that specifc image (assuming that the image contains
        a shurikode encoded code).

        :param `img`: The Pillow Image or `torch.Tensor` (as a 3D tensor or a 4D tensor of batch size 1).
        """
        with torch.no_grad():
            img_t = self.__img_to_expected_tensor(img, self.__device)
            logits: Tensor = self.__m(img_t).squeeze(0)
            
            probs = torch.softmax(logits, dim=-1)
            label = int(probs.argmax(-1).item())
            confidence = float(probs.max().item())

            #just for debugging
            top3 = torch.topk(probs, k=3)
            top3_labels = top3.indices.tolist()
            top3_confs  = top3.values.tolist()
            print(f"Top 3 predictions: {list(zip(top3_labels, top3_confs))}")
            
        return label, confidence

    @staticmethod
    def __img_to_expected_tensor(img: Union[Image, Tensor], device: str) -> Tensor:
        assert device in ["cpu", "cuda", "mps"]

        import numpy as np
        from torchvision import transforms as T
        
        eval_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if isinstance(img, PIL.Image.Image):
            img_np = np.array(img.convert("RGB"))
        elif isinstance(img, Tensor):
            if len(img.shape) == 4:
                img = img.squeeze(0)
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        tensor = eval_transform(img_np).unsqueeze(0).to(device)
        return tensor
