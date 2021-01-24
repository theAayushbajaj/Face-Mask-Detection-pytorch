""" Add images into a pandas Dataframe
"""
from pathlib import Path

import pandas as pd
#from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdm import tqdm
import os
# download dataset from link provided by
# https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
# datasetPath = Path('data/mask.zip')
# gdd.download_file_from_google_drive(file_id='1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp',
#                                     dest_path=str(datasetPath),
#                                     unzip=True)
# # delete zip file
# datasetPath.unlink()

datasetPath = Path('/home/aayush/dataset')
maskPath = datasetPath/'with_mask'
nonMaskPath = datasetPath/'without_mask'
maskDF = pd.DataFrame()

for f in os.listdir(maskPath):
    if os.path.isfile(os.path.join(maskPath, f)):
        maskDF = maskDF.append({
                'image': str(os.path.join(maskPath,f)),
                'mask': 1
            }, ignore_index=True)

for f in os.listdir(nonMaskPath):
    if os.path.isfile(os.path.join(nonMaskPath, f)):
        maskDF = maskDF.append({
                'image': str(os.path.join(nonMaskPath,f)),
                'mask': 0
            }, ignore_index=True)

maskDF = maskDF.sample(frac=1).reset_index(drop=True)
dfName = 'metadata/mask_df.pickle'
print(f'saving Dataframe to: {dfName}')
maskDF.to_pickle(dfName)
print('With Mask',len(maskDF.loc[maskDF['mask']==1]))
print('Without Mask',len(maskDF.loc[maskDF['mask']==0]))
