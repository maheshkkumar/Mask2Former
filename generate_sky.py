import argparse
import os

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from PIL import Image
from tqdm import tqdm

from mask2former import add_maskformer2_config

coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file(
    "configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
predictor = DefaultPredictor(cfg)


def generate_masks(args):

    # read images
    images = sorted([os.path.join(args.image_path, x)
                    for x in os.listdir(args.image_path)])

    print(f"Processing {len(images)} images")

    os.makedirs(args.output_path, exist_ok=True)

    for img_path in tqdm(images, total=len(images)):
        img = cv2.imread(img_path)
        file_name = os.path.basename(img_path).split('.')[0]
        output_path = os.path.join(args.output_path, f"{file_name}.png")

        if os.path.exists(output_path):
            continue

        outputs = predictor(img)

        # save sky segmentation masks
        sem_seg = outputs["sem_seg"].argmax(0).cpu().numpy()
        binary_mask = (sem_seg == args.segm_label).astype(np.uint8) * 255.

        Image.fromarray(binary_mask).convert("L").save(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path of the images')
    parser.add_argument('--output_path', type=str, help='Path of the output')
    parser.add_argument('--segm_label', default=119,
                        type=int, help='Label of the segmentation, default is 119 (sky)')
    args = parser.parse_args()
    generate_masks(args)
