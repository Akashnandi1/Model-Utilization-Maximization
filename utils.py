import os
import pandas as pd
from config import *
import cv2
from tqdm import tqdm
import logging
import json

# create dataframe of train and val
def df_creator(path, cls_chosen, split_data):
	data = {}
	i = 0
	for cls in cls_chosen:
		for file in os.listdir(path+cls):
			flag = True
			if file in split_data[cls]['train']:
				split = 'train'
			elif file in split_data[cls]['test']:
				split = 'test'
			else:
				#print(file, cls, 'file not found in split_data')
				flag = False
			file_path = path+cls+'/'+file
			
			if flag:
				data[i] = {'class': cls, 'split': split, 'path': file_path, 'filename': file}
				i += 1
			
	df = pd.DataFrame.from_dict(data, "index")			
			
	return df

def file_names(content, split_type, data, cls_chosen):
	counter = 0
	for item in content:
		folder = item.split('/')[1]
		file = item.split('/')[-1].split(' ')[0]
		if folder in cls_chosen:
			counter+=1
			if folder not in data:
				data[folder] = {'train': [], 'test': []}
				data[folder][split_type].append(file)
			else:
				data[folder][split_type].append(file)
	logging.info(str(counter) + ' ' + 'files added to ' +  str(split_type))

def DomainNet_select_classes(sample_threshold):
	c_p = DATASET+'/'+'clipart/'
	p_p = DATASET+'/'+'painting/'
	r_p = DATASET+'/'+'real/'
	s_p = DATASET+'/'+'sketch/'

	c_cls = set(os.listdir(c_p))
	p_cls = set(os.listdir(p_p))
	r_cls = set(os.listdir(r_p))
	s_cls = set(os.listdir(s_p))

	cls_chosen = c_cls.intersection(p_cls).intersection(r_cls).intersection(s_cls)

	info = {}
	SAMPLE_THRESHOLD = 165
	for item in sorted(cls_chosen):
		temp = {'c': 0, 'p': 0, 'r': 0, 's':0}
		temp['c'] = len(os.listdir(c_p+item))
		temp['p'] = len(os.listdir(p_p+item))
		temp['r'] = len(os.listdir(r_p+item))
		temp['s'] = len(os.listdir(s_p+item))
		if temp['c'] >= SAMPLE_THRESHOLD and temp['p'] >= SAMPLE_THRESHOLD and temp['r'] >= SAMPLE_THRESHOLD and temp['s'] >= SAMPLE_THRESHOLD:
			info[item] = temp

	return info

def create_DomainNet_data(SRC, ADP):
	DATASET = './datasets/DomainNet'
	SRC_SPLIT_TRAIN_PATH = DATASET+'/'+SRC+'_train.txt'
	SRC_SPLIT_TEST_PATH = DATASET+'/'+SRC+'_test.txt'
	ADP_SPLIT_TEST_PATH = DATASET+'/'+ADP+'_test.txt'
	ADP_SPLIT_TRAIN_PATH = DATASET+'/'+ADP+'_train.txt'
	SAMPLE_THRESHOLD = 165
	PATH_SRC = DATASET+'/'+SRC+'/'
	PATH_ADP = DATASET+'/'+ADP+'/'

	# Select common pool of classes with more than sample_threshold images in each

	info = DomainNet_select_classes(SAMPLE_THRESHOLD)
	cls_chosen = list(info.keys())

	## Selection complete

	with open(SRC_SPLIT_TEST_PATH) as test_src:
		src_contents_test = test_src.readlines()
		
	with open(SRC_SPLIT_TRAIN_PATH) as train_src:
		src_contents_train = train_src.readlines()

	with open(ADP_SPLIT_TEST_PATH) as test_adp:
		adp_contents_test = test_adp.readlines()

	with open(ADP_SPLIT_TRAIN_PATH) as train_adp:
		adp_contents_train = train_adp.readlines()

	# path_src = PATH_SRC
	# path_adp = PATH_ADP
	# cls_src = set(os.listdir(path_src))
	# cls_adp = set(os.listdir(path_adp))
	# cls_chosen = cls_src.intersection(cls_adp)
	#
	#
	# info = {}
	# for item in sorted(cls_chosen):
	# 	temp = {'src': 0, 'adp': 0}
	# 	temp['src'] = len(os.listdir(path_src+item))
	# 	temp['adp'] = len(os.listdir(path_adp+item))
	# 	if temp['src'] >= SAMPLE_THRESHOLD and temp['adp'] >= SAMPLE_THRESHOLD:
	# 		info[item] = temp

	total_classes = len(cls_chosen)

	src_split_data = {}
	adp_split_data = {}

	file_names(src_contents_train, 'train', src_split_data, cls_chosen)
	file_names(src_contents_test, 'test', src_split_data, cls_chosen)

	file_names(adp_contents_train, 'train', adp_split_data, cls_chosen)
	file_names(adp_contents_test, 'test', adp_split_data, cls_chosen)

	src_df = df_creator(PATH_SRC, cls_chosen, src_split_data)
	aug_df = df_creator(PATH_ADP, cls_chosen, adp_split_data)
	src_df['label'] = None
	aug_df['label'] = None

	label_value = {}
	counter = 0
	for cls in cls_chosen:
		label_value[cls] = counter
		src_df.loc[src_df['class'] == cls, 'label'] = counter
		aug_df.loc[aug_df['class'] == cls, 'label'] = counter
		counter += 1

	# with open(cfg['exp']['log_folder'] + '/keys.json', 'w') as fp:
	# 	json.dump(label_value, fp)

	src_train_df = src_df[src_df['split'] == 'train']
	src_test_df = src_df[src_df['split'] == 'test']
	
	src_train_df.reset_index(inplace = True)
	src_test_df.reset_index(inplace = True)
	
	src_train_df.drop('index', axis =1, inplace = True)
	src_test_df.drop('index', axis = 1, inplace = True)

	adp_train_df = aug_df[aug_df['split'] == 'train']
	adp_test_df = aug_df[aug_df['split'] == 'test']
	
	adp_train_df.reset_index(inplace = True)
	adp_test_df.reset_index(inplace = True)
	
	adp_train_df.drop('index', axis =1, inplace = True)
	adp_test_df.drop('index', axis = 1, inplace = True)

	src_train_df = src_train_df.sample(frac = 1)
	src_test_df = src_test_df.sample(frac = 1)
	adp_train_df = adp_train_df.sample(frac = 1)
	adp_test_df = adp_test_df.sample(frac = 1)

	return src_train_df, src_test_df, adp_train_df, adp_test_df, total_classes

def create_aug_images(train_df, augment_folder_path, aug):
	for index in tqdm(train_df.index):

		filename = train_df['filename'][index]
		cls = train_df['class'][index]
		path = train_df['path'][index]

		img = cv2.imread(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


		cls_path = augment_folder_path + cls
		if cls not in os.listdir(augment_folder_path):
			os.mkdir(cls_path)
		os.mkdir(cls_path + '/' + filename)
		for i in range(AUG_COUNT):
			out = aug(images = [img])
			out = cv2.resize(out[0], (IMG_SIZE, IMG_SIZE))
			final_path = cls_path + '/' + filename + '/' + str(i) + '.jpg'
			cv2.imwrite(final_path, out)

def aug_df_creator(train_df, augment_folder_path):
	data = {}
	i = 0
	for index in train_df.index:
		cls = train_df['class'][index]
		file_name = train_df['filename'][index]
		aug_cls_path = augment_folder_path + cls + '/' + file_name + '/'
		label = train_df['label'][index]
		for item in os.listdir(aug_cls_path):
			path = aug_cls_path + item
			data[i] = {'class': cls, 'parent_filename': file_name, 'aug_filename': item, 'path': path, 'label': label}
			i += 1
	df = pd.DataFrame.from_dict(data, "index")
	return df

def model_inp_size(arch):
	if arch == 'eb0':
		img_size = 224
	elif arch == 'eb1':
		img_size = 240
	elif arch == 'eb2':
		img_size = 260
	elif arch == 'eb3':
		img_size = 300
	else:
		raise Exception('Model architecture not implemented.')
	return img_size