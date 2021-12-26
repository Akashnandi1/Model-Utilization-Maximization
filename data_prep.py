import pandas as pd
from utils import *
from imgaug import augmenters as iaa
from config import *

augment_folder = ADP +'-augment'
augment_folder_path = DATASET + '/' + augment_folder + '/'
src_train_df, src_test_df, adp_train_df, adp_test_df = create_data()

create_aug_dataset = True
if augment_folder not in os.listdir(DATASET):
    os.mkdir(augment_folder_path)
else:
    create_aug_dataset = False

if create_aug_dataset:
    aug = iaa.RandAugment(n=3, m=2)
    create_aug_images(adp_train_df, augment_folder_path, aug)

aug_df = aug_df_creator(adp_train_df, augment_folder_path)

aug_df.to_csv(DATASET+'/aug_df_' + ADP + '.csv')
adp_train_df.to_csv(DATASET+'/src_train_df_' + ADP + '.csv')
adp_test_df.to_csv(DATASET+'/src_test_df_' + ADP + '.csv')
src_train_df.to_csv(DATASET+'/src_train_df_' + SRC + '.csv')
src_test_df.to_csv(DATASET+'/src_test_df_' + SRC + '.csv')