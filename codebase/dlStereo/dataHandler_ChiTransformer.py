
import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

class ChiTransformer_StereoDepth(): 
    def __init__(self, db_fp = 'C:/Users/nisar2/Desktop/final_543/inferences',prefix = 'result_color_', ext = '.pfm' ):

        self.pfmFiles   = [f for f in os.listdir(db_fp) if f.endswith(ext)]
        self.num_frames = len(self.pfmFiles)
        self.db_fp      = db_fp
        self.prefix     = prefix
        self.ext        = ext
    
    def get_depthMap(self,img_num):
        file_path = f'{self.db_fp}/{self.prefix}{self.map_to_zeros(img_num)}{self.ext}'
        print('DL handler: ',file_path)
        dl_depth_map = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        dl_depth_map = cv2.resize(dl_depth_map, (1241,376), interpolation=cv2.INTER_CUBIC)
        return dl_depth_map
    
    def map_to_zeros(self,number, width=6):
        # Convert the number to a string and fill with zeros to the specified width
        result = str(number).zfill(width)
        return result
    
if __name__ == "__main__":
    handler = ChiTransformer_StereoDepth()

    # Examples
    for i in range(5):  # Replace 5 with your desired range
        dl_depth = handler.get_depthMap(img_num=i)
        print(dl_depth.shape)
        plt.imshow(dl_depth)
        plt.show()



