import torch 
import torch.utils.data as data
from src.core import register

__all__ = ['DataLoader']

@register
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

@register
def default_collate_fn(items):
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]

@register
def augment_data(image, boxes):
    small_boxes = [box for box in boxes if (box[2] - box[0]) * (box[3] - box[1]) < 32 * 32]
    if small_boxes:
        image = image.crop((small_boxes[0][0], small_boxes[0][1], small_boxes[0][2], small_boxes[0][3]))
    image = image.resize((512, 512))
    return image
