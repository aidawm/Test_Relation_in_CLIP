from transformers import CLIPProcessor, CLIPModel
from dataclasses import dataclass

class CLIPConfig:
    def __init__(self, image_size: int = 336, patch_size: int = 14, device = "cpu", model_link: str= "openai/clip-vit-large-patch14-336", which_function: int = 1, num_layers: int = 24 ):
      
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size: int = image_size // patch_size
        self.num_patches: int = self.grid_size * self.grid_size
        self.device = device
        self.model_link = model_link
        self.model = CLIPModel.from_pretrained(model_link).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_link)
        self.which_function=  which_function
        self.num_layers = num_layers

    

