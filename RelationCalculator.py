import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw, ImageFont
import math
import VisualRelationshipDetection_Dataset as vrd
from Configs import CLIPConfig




class Relation_Calculator: 
    def __init__ (self, image, h_l_pair ,  config:CLIPConfig):
        self.config = config
        self.model = self.config.model
        self.processor = self.config.processor
        self.image = image
        # self.layer = layer
        # self.head = head
        self.layer_head_pair_list = h_l_pair
        self.__get_last_layer_attention()
        
        
            
    def __get_last_layer_attention (self):
        inputs = self.processor(
            text=["a photo"], images=self.image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()} 
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention from the last layer of the vision transformer
        self.vision_attentions = outputs.vision_model_output.attentions  # list of layers
        self.last_layer_attention = torch.stack([self.vision_attentions[l][0][h] for l,h in self.layer_head_pair_list])
        # self.last_layer_attention = self.vision_attentions[self.layer][0][self.head]  # shape: (1, num_heads, seq_len, seq_len)
    
    def process_attention(self, attention_values):
        if self.config.which_function == 0:
            return attention_values.mean(dim=(1,2)).mean().item()  # Mean over heads and tokens
        elif self.config.which_function == 1:
            return attention_values.mean(dim=1).max(dim=1).values.mean().item() 
        elif self.config.which_function == 2:
            return attention_values.mean(dim=(1,2)).max().item()
        elif self.config.which_function == 3:
            return attention_values.max(dim=1).values.mean(dim=1).max().item()
        elif self.config.which_function == 4:
            return attention_values.mean(dim=1).mean(dim=1).prod(dim=0) 

    def get_attention_between_objects(self, attention, object1_patch_list, object2_patch_list, has_cls_token=True):
        """
        attention: torch.Tensor of shape (num_heads, num_tokens, num_tokens)
        object1_patch_list: list of ints (patch indices of object 1)
        object2_patch_list: list of ints (patch indices of object 2)
        has_cls_token: whether attention map includes CLS token at index 0
        """
        offset = 1 if has_cls_token else 0
        # Shift patch indices if CLS token is present
        object1_indices = [i + offset for i in object1_patch_list]
        object2_indices = [j + offset for j in object2_patch_list]

        # Extract attention values from object1 patches to object2 patches
        att_values = attention[:,object1_indices, :][:,:, object2_indices]  # [heads, len(obj1), len(obj2)]
      

        return att_values.mean(dim=1).max(dim=1).values.max().item()
      
    def get_relation(self, bbox1,bbox2):
        patches1 = self.get_patch_indices_in_bbox(bbox1)
        patches2 = self.get_patch_indices_in_bbox(bbox2)
        return self.get_attention_between_objects(self.last_layer_attention, patches1, patches2)


    def get_patch_indices_in_bbox(self, bbox):
        patch_size = self.config.image_size // self.config.grid_size  

        xmin, ymin, xmax, ymax = bbox
        # Convert pixel coordinates to patch indices
        left = math.floor(xmin / patch_size)
        right = math.floor((xmax - 1) / patch_size)
        top = math.floor(ymin / patch_size)
        bottom = math.floor((ymax - 1) / patch_size)

         # Clamp to valid grid range
        left = max(0, min(left, self.config.grid_size - 1))
        right = max(0, min(right, self.config.grid_size - 1))
        top = max(0, min(top, self.config.grid_size - 1))
        bottom = max(0, min(bottom, self.config.grid_size - 1))

        patch_indices = []
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                patch_index = i * self.config.grid_size   + j  # flatten to 1D index
                patch_indices.append(patch_index)
        return patch_indices

    