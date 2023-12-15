import os
import glob
import torch
import cv2
import argparse
from functools import partial
from tqdm import tqdm 

from torchvision.transforms import Compose

import sys
sys.path.append('../')
from ChiTransformer.model.chitransformer import ChitransformerDepth
from ChiTransformer.model.dcr import DepthCueRectification_Sp
from ChiTransformer.utils.inference_utils import *
import numpy as np

def run_inference(model, input_path, output_path, model_path=None, optimize=True, kitti_crop=True):
    ''' 
    Reference: https://github.com/isl-cv/chitransformer
    Inference pipeline to produce depth maps using ChiTransformer Model Approach.

    Args:
        input_path (str): Path to the input images.
        output_path (str): Path to save the output depth maps.
        model_path (str, optional): Path to the pre-trained model. Defaults to None.
        optimize (bool, optional): Flag to enable model optimization. Defaults to True.
        kitti_crop (bool, optional): Flag to enable KITTI cropping. Defaults to False.

    Return:
        None
    '''

    net_w = 1216
    net_h = 352

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Running ChiTransformer Inference:')
    print('\n---- ChiTransformer Configuration ----')
    print(f'input-path:\t{input_path}')
    print(f'output-path:\t{output_path}')
    print(f'model-path:\t{model_path}')
    print(f'optimize:\t{optimize}')
    print(f'kitti_crop:\t{kitti_crop}')
    print(f"device:\t{device}")
    print('-------------------------------------\n')
    
    

    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target      = None,
                    keep_aspect_ratio  = True,
                    ensure_multiple_of = 32,
                    resize_method="minimal",
                    image_interpolation_method = cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format = torch.channels_last)
        model = model.half()

    model.to(device)

    # get input
    assert os.path.isdir(os.path.join(input_path, "image_left")) and \
    os.path.isdir(os.path.join(input_path, "image_right")),\
    'Put left and right images in folder /image_left and /image_right respectively.'

    image_names = os.listdir(input_path+"/image_left/")

    img_list_left = [input_path+"/image_left/"+name for name in image_names]

    num_images = len(img_list_left)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    for ind, img_left in enumerate(img_list_left):
        if os.path.isdir(img_left):
            img_fmt = img_left.split(".")[-1]
            if img_fmt in ['jpg', 'png']:
                continue
            
        name = img_left.split("/")[-1]

        img_right = os.path.join(input_path, 'image_right', name)
        if os.path.isdir(img_right):
            continue

        # print("  processing {} and {} ({}/{})".format(img_left, img_right, ind + 1, num_images))
        # input

        img_left = read_image(img_left)
        img_right = read_image(img_right)

        if kitti_crop is True:
            height, width, _ = img_left.shape
            top = height - 352
            left = (width - 1216) // 2
            img_left = img_left[top : top + 352, left : left + 1216, :]
            img_right = img_right[top : top + 352, left : left + 1216, :]

        img_left_input = transform({"image": img_left})["image"]
        img_right_input = transform({"image": img_right})["image"]

        # compute
        
        with torch.no_grad():
            img_left_input = torch.from_numpy(img_left_input).to(device).unsqueeze(0)
            img_right_input = torch.from_numpy(img_right_input).to(device).unsqueeze(0)
            
            if optimize == True and device == torch.device("cuda"):
                img_left_input  = img_left_input.to(memory_format=torch.channels_last)
                img_right_input = img_right_input.to(memory_format=torch.channels_last)
                img_left_input  = img_left_input.half()
                img_right_input = img_right_input.half()
                
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model.forward(img_left_input, img_right_input)
            else:
                    prediction = model.forward(img_left_input, img_right_input)

            prediction = prediction[("depth", 0)]
            prediction = (
                            torch.nn.functional.interpolate(
                                prediction.unsqueeze(1),
                                size=img_left.shape[:2],
                                mode="bicubic",
                                align_corners=False,
                            )
                            .squeeze()
                            .cpu()
                            .numpy()
                        )

        filename = os.path.join(
                    output_path,
            os.path.splitext(os.path.basename(f'result_color_' + name.split('.')[0] + '.png'))[0]
                )
        write_depth_color(filename, 1/prediction+1e-8, absolute_depth=True)

    print("ChiTransformer Inference complete! ")



if __name__ == "__main__":
    model_fp    = 'C:/Users/nisar2//Desktop/final_543/database/chitransformer_kitti_15_03.pth'
    input_path  = 'C:/Users/nisar2/Desktop/final_543/database/images'
    output_path = 'C:/Users/nisar2/Desktop/final_543/infer_temp'
    sequences   = ['00','01', '02', '03', '04', '05']
    bool_kitti_crop = True

    print('Running Inference of ChiTransformer on Image Pairs\n')
    device = torch.device("cuda")
    model = ChitransformerDepth( 
                                    device=device, 
                                    dcr_module=partial(DepthCueRectification_Sp, 
                                    layer_norm=False)
                                    ).to(device)
    if model_fp:
        checkpoint = torch.load(model_fp, map_location='cpu')
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            model.load_state_dict(checkpoint)

    for i in range(len(sequences)):

        sequence = sequences[i]
        seq_input_path = f'{input_path}/{sequence}'
        o_path         = f'{output_path}/{sequence}'

        # # compute depth maps using ChiTransformer DL Model  
        run_inference(
            model,
            input_path  = seq_input_path,
            output_path= o_path,
            model_path = model_fp,
            optimize= True,
            kitti_crop= bool_kitti_crop,
        )
