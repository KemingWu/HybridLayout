from typing import Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os

def scale_boxes(boxes, width, height):
    scaled_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        scaled_box = [x_min * width, y_min * height, x_max * width, y_max * height]
        scaled_boxes.append(scaled_box)
    return scaled_boxes

def draw_mask(mask, draw, random_color=True):
    if random_color:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            153,
        )
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))
    
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)
        
def bbox_visualization(image_pil: Image,
                        result: Dict,
                        draw_width: float = 6.0,
                        return_mask=True) -> Image:
    """Plot bounding boxes and labels on an image.

    Args:
        image_pil (PIL.Image): The input image as a PIL Image object.
        result (Dict[str, Union[torch.Tensor, List[torch.Tensor]]]): The target dictionary containing
            the bounding boxes and labels. The keys are:
                - boxes (List[int]): A list of bounding boxes in shape (N, 4), [x1, y1, x2, y2] format.
                - scores (List[float]): A list of scores for each bounding box. shape (N)
                - labels (List[str]): A list of labels for each object
                - masks (List[PIL.Image]): A list of masks in the format of PIL.Image
        draw_score (bool): Draw score on the image. Defaults to False.

    Returns:
        PIL.Image: The input image with plotted bounding boxes, labels, and masks.
    """
    # Get the bounding boxes and labels from the target dictionary
    boxes = result["boxes"]
    categorys = result["labels"]
    masks = result.get("masks", [])

    
    color_list= [(177, 214, 144),(255, 162, 76),
                (13, 146, 244),(249, 84, 84),(54, 186, 152),
                (74, 36, 157),(0, 159, 189),
                (80, 118, 135),(188, 90, 148),(119, 205, 255)]


    np.random.seed(42)

    # Find all unique categories and build a cate2color dictionary
    cate2color = {}
    unique_categorys = sorted(set(categorys))
    for idx,cate in enumerate(unique_categorys):
        cate2color[cate] = color_list[idx%len(color_list)]
    
    # Load a font with the specified size
    font_size=30
    # without using specific font and set font size
    font = ImageFont.load_default()
    font = font.font_variant(size=font_size)


    # Create a PIL ImageDraw object to draw on the input image
    if isinstance(image_pil, np.ndarray):
        image_pil = Image.fromarray(image_pil)
    draw = ImageDraw.Draw(image_pil)
    
    # Create a new binary mask image with the same size as the input image
    mask = Image.new("L", image_pil.size, 0)
    # Create a PIL ImageDraw object to draw on the mask image
    mask_draw = ImageDraw.Draw(mask)

    # Draw boxes, labels, and masks for each box and label in the target dictionary
    for box, category in zip(boxes, categorys):
        # Extract the box coordinates
        x0, y0, x1, y1 = box

        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        color = cate2color[category]

        # Draw the box outline on the input image
        draw.rectangle([x0, y0, x1, y1], outline=color, width=int(draw_width))

        # Draw the label and score on the input image
        text = f"{category}"

        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), text, font)
        else:
            w, h = draw.textsize(text, font)
            bbox = (x0, y0, w + x0, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), text, fill="white",font=font)

    # Draw the mask on the input image if masks are provided
    if len(masks) > 0 and return_mask:
        size = image_pil.size
        mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            mask = np.array(mask)[:, :, -1]
            draw_mask(mask, mask_draw)

        image_pil = Image.alpha_composite(image_pil.convert("RGBA"), mask_image).convert("RGB")
    return image_pil
