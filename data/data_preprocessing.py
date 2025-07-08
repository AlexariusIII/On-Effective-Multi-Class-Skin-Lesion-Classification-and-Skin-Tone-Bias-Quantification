# %%
import pandas as pd
import json
import os
import numpy as np
import ast

# %%
data_dir = '../../data'

# Data with fp information
fitzpatrick_src = os.path.join(data_dir, 'fritzpatrick/images/huggingface_fritzpatrick')
ddi_src = os.path.join(data_dir, 'ddi/images')
pad_ufes_20_src = os.path.join(data_dir, 'pad-ufes-20/images')
scin_src = os.path.join(data_dir, 'scin/images')
mskcc_src = os.path.join(data_dir, 'MSKCC/images')
argentina_src = os.path.join(data_dir,'argentina/images')

# Data without fp information
isic_src = os.path.join(data_dir,"isic_24")
ham_src = os.path.join(data_dir,"ham10000")
bcn_src = os.path.join(data_dir,"bcn20000")
derm12345_src = os.path.join(data_dir,"derm12345")
derm7pt_src = os.path.join(data_dir,"derm7pt")
med_node_src = os.path.join(data_dir,"med_node")
sd_198_src = os.path.join(data_dir,"sd-198")

# %%
binary_mapping = {
    'ACK': 'benignant',
    'BCC': 'malignant',
    'MEL': 'malignant',
    'NEV': 'benignant',
    'SCC': 'malignant',
    'SEK': 'benignant'
}

class_mapping = json.load(open('class_mapping.json'))
processed_dfs = []

# %% [markdown]
# ## Fitzpatrick17K

# %%
fp_metadata_path = os.path.join(data_dir, 'fritzpatrick/fitzpatrick17k.csv')
fp_metadata = pd.read_csv(fp_metadata_path)
print(len(fp_metadata))
fp_metadata.head()

# %%
# Map skin tone
fp_scale_mapping = {
    1: 'A',
    2: 'A',
    3: 'B',
    4: 'B',
    5: 'C',
    6: 'C',
    
}
fp_dataframe = fp_metadata[['md5hash', 'fitzpatrick_scale', 'label', 'nine_partition_label', 'three_partition_label']].copy()
fp_dataframe.rename(columns={'md5hash': 'image_id', 'fitzpatrick_scale': 'skin_tone', 'label': 'disease', 'nine_partition_label': 'nine_partition', 'three_partition_label': 'three_partition'}, inplace=True)
fp_dataframe['skin_tone'] = fp_dataframe['skin_tone'].map(fp_scale_mapping)
#add .jpg to image_id
fp_dataframe['image_id'] = fp_dataframe['image_id'].apply(lambda x: x + '.jpg')
#drop skin tone nan values
print(len(fp_dataframe))
fp_dataframe.head()

# %%
# Map multiclass targets
fp_df_processed = fp_dataframe.copy()
fp_df_processed['multiclass'] = fp_df_processed['disease'].map(class_mapping)
fp_df_processed['binary'] = fp_df_processed['multiclass'].map(binary_mapping)
fp_df_processed['dataset'] = 'fp'
fp_df_processed['image_type']='clinical'
fp_df_processed = fp_df_processed.drop_duplicates(subset=['image_id'])
fp_df_processed = fp_df_processed[['image_id', 'binary', 'multiclass', 'skin_tone', 'dataset','image_type']]

fp_df_processed.to_csv('processed_data/fp_data.csv', index=False)
processed_dfs.append(fp_df_processed)
print(len(fp_df_processed))
fp_df_processed.head()

# %% [markdown]
# ## DDI

# %%
ddi_metadata_path = os.path.join(data_dir, 'ddi/ddi_metadata.csv')
ddi_metadata = pd.read_csv(ddi_metadata_path)
print(len(ddi_metadata))
ddi_metadata.head()

# %%
# Map skin tone
ddi_skintone_mapping = {12: 'A', 34: 'B', 56: 'C'}
ddi_dataframe = ddi_metadata[['DDI_file', 'skin_tone', 'disease', 'malignant']].copy()
ddi_dataframe['skin_tone'] = ddi_dataframe['skin_tone'].map(ddi_skintone_mapping)
#rename columns
ddi_dataframe.rename(columns={'DDI_file': 'image_id', 'skin_tone': 'skin_tone', 'disease': 'disease', 'malignant': 'malignant'}, inplace=True)
print(len(ddi_dataframe))
ddi_dataframe.head()

# %%
# Map multiclass targets
ddi_df_processed = ddi_dataframe.copy()
ddi_df_processed['multiclass'] = ddi_df_processed['disease'].map(class_mapping)
ddi_df_processed['binary'] = ddi_df_processed['multiclass'].map(binary_mapping)
ddi_df_processed['dataset'] = 'ddi'
ddi_df_processed['image_type']='clinical'
ddi_df_processed = ddi_df_processed.drop_duplicates(subset=['image_id'])

ddi_df_processed = ddi_df_processed[['image_id', 'binary', 'multiclass', 'skin_tone', 'dataset','image_type']]
ddi_df_processed.to_csv('processed_data/ddi_data.csv', index=False)
processed_dfs.append(ddi_df_processed)
print(len(ddi_df_processed))
ddi_df_processed.head()

# %% [markdown]
# ## PAD-UFES-20

# %%
pad_metadata_path = os.path.join(data_dir, 'pad-ufes-20/metadata.csv')
pad_metadata = pd.read_csv(pad_metadata_path)
print(len(pad_metadata))
pad_metadata.head()

# %%
# Map skin tone
pad_skin_tone_mapping = {
    1.0: 'A',
    2.0: 'A',
    3.0: 'B',
    4.0: 'B',
    5.0: 'C',
    6.0: 'C',
}
pad_dataframe = pad_metadata[['img_id', 'fitspatrick', 'diagnostic']].copy()
#rename columns
pad_dataframe.rename(columns={'img_id': 'image_id', 'fitspatrick': 'skin_tone', 'diagnostic': 'disease'}, inplace=True)
pad_dataframe['skin_tone'] = pad_dataframe['skin_tone'].map(pad_skin_tone_mapping)
print(len(pad_dataframe))
pad_dataframe.head()

# %%
# Map multiclass targets
pad_df_processed = pad_dataframe.copy()

pad_df_processed.rename(columns={'disease': 'multiclass'}, inplace=True)
pad_df_processed['binary']=pad_df_processed['multiclass'].map(binary_mapping)
pad_df_processed['dataset']='pad'
pad_df_processed['image_type']='clinical'
pad_df_processed = pad_df_processed.drop_duplicates(subset=['image_id'])

pad_df_processed = pad_df_processed[['image_id', 'binary', 'multiclass', 'skin_tone', 'dataset','image_type']]
pad_df_processed.to_csv('processed_data/pad_data.csv', index=False)
processed_dfs.append(pad_df_processed)
print(len(pad_df_processed))
pad_df_processed.head()

# %% [markdown]
# ## SCIN

# %%
scin_metadata_path = os.path.join(data_dir, 'scin/scin_data.csv')
scin_metadata = pd.read_csv(scin_metadata_path)
print(len(scin_metadata))
scin_metadata.head()

# %%
# out of dermatologist_fitzpatrick_skin_type_label_1,2,3 create a new column with the most common value return nan if all are different
def get_consensus(row):
    values = row.dropna()
    mode = values.mode()
    # If exactly one mode, return it; else, return NaN
    if len(mode) == 1:
        return mode.iloc[0]
    else:
        return np.nan

scin_dataframe = scin_metadata.copy()
scin_dataframe['dermatologist_fitzpatrick_skin_type_label_consense'] = scin_metadata[
    ['dermatologist_fitzpatrick_skin_type_label_1',
     'dermatologist_fitzpatrick_skin_type_label_2',
     'dermatologist_fitzpatrick_skin_type_label_3']
].apply(get_consensus, axis=1)

for index, row in scin_dataframe.iterrows():
    if row['image_1_shot_type'] == 'CLOSE_UP':
        #use the image_1_path as image_id
        scin_dataframe.at[index, 'image_id'] = row['image_1_path']
    elif row['image_2_shot_type'] == 'CLOSE_UP':
        #use the image_2_path as image_id
        scin_dataframe.at[index, 'image_id'] = row['image_2_path']
    elif row['image_3_shot_type'] == 'CLOSE_UP':
        #use the image_3_path as image_id
        scin_dataframe.at[index, 'image_id'] = row['image_3_path']
    else:
        scin_dataframe.at[index, 'image_id'] = None

scin_dataframe['disease'] = scin_dataframe['dermatologist_skin_condition_on_label_name'].apply(ast.literal_eval)
scin_dataframe = scin_dataframe.explode(['disease'])

print(len(scin_dataframe))
scin_dataframe.head()


# %%
# Map skin tone
scin_skin_tone_mapping = {
    'FST1': 'A',
    'FST2': 'A',
    'FST3': 'B',
    'FST4': 'B',
    'FST5': 'C',
    'FST6': 'C',
}

scin_dataframe['skin_tone'] = scin_dataframe['dermatologist_fitzpatrick_skin_type_label_consense'].map(scin_skin_tone_mapping)
#drop skin tone nan values
print(len(scin_dataframe))
scin_dataframe.head()

# %%
#Map multiclass target
scin_df_processed = scin_dataframe.copy()
#scin_df_processed = scin_df_processed.drop_duplicates(subset=['image_id'])
scin_df_processed['multiclass'] = scin_df_processed['disease'].map(class_mapping)
scin_df_processed['binary'] = scin_df_processed['multiclass'].map(binary_mapping)
scin_df_processed['dataset'] = 'scin'
scin_df_processed['image_type']='clinical'
#drop rows with nan image id
scin_df_processed = scin_df_processed[scin_df_processed['image_id'].notna()].copy()
#only use base of image_id
scin_df_processed['image_id'] = scin_df_processed['image_id'].str.split('/').str[-1]

scin_df_processed = scin_df_processed[['image_id', 'binary', 'multiclass', 'skin_tone', 'dataset','image_type']]
scin_df_processed.to_csv('processed_data/scin_data.csv', index=False)

processed_dfs.append(scin_df_processed)
print(len(scin_df_processed))
scin_df_processed.head()

# %% [markdown]
# ## MSKCC

# %%
mskcc_metadata_path = os.path.join(data_dir, 'MSKCC/metadata.csv')
mskcc_metadata = pd.read_csv(mskcc_metadata_path)
print(len(mskcc_metadata))
mskcc_metadata.head()

# %%
# Map skin tone

mskcc_skintone_mapping = {'I': 'A','II': 'A', 'III': 'B', 'VI': 'B','V': 'C','VI': 'C'}

mskcc_dataframe = mskcc_metadata[['isic_id', 'fitzpatrick_skin_type', 'diagnosis_1','image_type']].copy()
mskcc_dataframe.rename(columns={'diagnosis_1': 'disease'}, inplace=True)
mskcc_dataframe['skin_tone'] = mskcc_dataframe['fitzpatrick_skin_type'].map(mskcc_skintone_mapping)
mskcc_dataframe['image_id'] = mskcc_dataframe['isic_id'].map(lambda x: x+'.jpg')

mskcc_dataframe = mskcc_dataframe[['image_id','skin_tone','disease','image_type']]
print(len(mskcc_dataframe))
mskcc_dataframe.head()

# %%
mskcc_df_processed = mskcc_dataframe.copy()
mskcc_df_processed['binary'] = mskcc_df_processed['disease'].map({"Benign": "benignant"})
mskcc_df_processed['dataset'] = 'mskcc'
mskcc_df_processed['multiclass'] = None
mskcc_df_processed['image_type']=mskcc_df_processed['image_type'].map(lambda x : 'dermatoscopic' if x== 'dermoscopic' else 'clinical')
mskcc_df_processed = mskcc_df_processed.drop_duplicates(subset=['image_id'])

mskcc_df_processed = mskcc_df_processed[['image_id', 'binary', 'multiclass', 'skin_tone', 'dataset','image_type']]
mskcc_df_processed.to_csv('processed_data/mskcc_data.csv', index=False)
processed_dfs.append(mskcc_df_processed)
print(len(mskcc_df_processed))
mskcc_df_processed.head()

# %% [markdown]
# ## HIBA

# %%
argentina_metadata_path = os.path.join(data_dir, 'argentina/metadata.csv')
argentina_metadata = pd.read_csv(argentina_metadata_path)
print(len(argentina_metadata))
argentina_metadata.head()

# %%
# Map Skin Tone
argentina_skintone_mapping = {'I': 'A','II': 'A', 'III': 'B', 'VI': 'B','V': 'C','VI': 'C'}

argentina_dataframe = argentina_metadata[['isic_id', 'fitzpatrick_skin_type', 'diagnosis','benign_malignant','image_type']].copy()
argentina_dataframe['skin_tone'] = argentina_dataframe['fitzpatrick_skin_type'].map(argentina_skintone_mapping)
argentina_dataframe['image_id'] = argentina_dataframe['isic_id'].map(lambda x: x+'.jpg')
argentina_dataframe.rename(columns={'diagnosis': 'disease'}, inplace=True)

argentina_dataframe = argentina_dataframe[['image_id','skin_tone','disease','benign_malignant','image_type']]
print(len(argentina_dataframe))
argentina_dataframe.head()

# %%
# Map multiclass target
argentina_df_processed = argentina_dataframe.copy()
argentina_df_processed['multiclass'] = argentina_df_processed['disease'].map(class_mapping)
argentina_df_processed['binary'] = argentina_df_processed['multiclass'].map(binary_mapping)
argentina_df_processed['dataset'] = 'argentina'
argentina_df_processed['image_type']=argentina_df_processed['image_type'].map(lambda x : 'dermatoscopic' if x== 'dermoscopic' else 'clinical')
argentina_df_processed = argentina_df_processed.drop_duplicates(subset=['image_id'])

argentina_df_processed = argentina_df_processed[['image_id', 'binary', 'multiclass', 'skin_tone', 'dataset','image_type']]

argentina_df_processed.to_csv('processed_data/argentina_data.csv', index=False)
processed_dfs.append(argentina_df_processed)
print(len(argentina_df_processed))
argentina_df_processed.head()

# %% [markdown]
# ## ISIC24

# %%
isic_metadata = pd.read_csv(os.path.join(isic_src,'train-metadata.csv'))
print(len(isic_metadata))
isic_metadata.head()

# %%
# Map multiclass target
isic_data = isic_metadata.copy()
isic_data['multiclass']=isic_data['iddx_full'].map(class_mapping)
isic_data['binary']= isic_data['target'].map({0:'benignant',1:'malignant'})
isic_data['dataset']='isic_24'
isic_data['image_id']=isic_data['isic_id'].map(lambda x: x+".jpg")
isic_data['image_type']='clinical'
isic_data = isic_data[isic_data['multiclass'].isna()==False].copy()
isic_data = isic_data[['image_id','binary','multiclass','dataset','image_type']]

processed_dfs.append(isic_data)
isic_data.to_csv('processed_data/isic_data.csv',index=False)
print(len(isic_data))
isic_data.head()

# %% [markdown]
# ## HAM10K

# %%
ham_metadata = pd.read_csv(os.path.join(ham_src,'HAM10000_metadata'))
print(len(ham_metadata))
ham_metadata.head()

# %%
# Map multiclass target
ham_data = ham_metadata.copy()
ham_data['multiclass']=ham_data['dx'].map(class_mapping)
ham_data['dataset']='ham'
ham_data['image_id']=ham_data['image_id'].map(lambda x: x+".jpg")
ham_data['binary']=ham_data['multiclass'].map(binary_mapping)
ham_data['image_type']='dermatoscopic'
ham_data = ham_data[ham_data['multiclass'].isna()==False].copy()
ham_data = ham_data[['image_id','binary','multiclass','dataset','image_type']]

#reduce nevi amount to 2000 images
ham_data_subset = ham_data[ham_data['multiclass'] == "NEV"]
sampled_target_rows = ham_data_subset.sample(n=2000, random_state=42)
ham_data_others = ham_data[ham_data['multiclass'] != "NEV"]
ham_data = pd.concat([ham_data_others, sampled_target_rows], ignore_index=True)

processed_dfs.append(ham_data)
ham_data.to_csv('processed_data/ham_data.csv',index=False)
print(len(ham_data))
ham_data.head()
ham_data.value_counts("multiclass",dropna=False)

# %% [markdown]
# ## BCN20K

# %%
bcn_metadata = pd.read_csv(os.path.join(bcn_src,'metadata.csv'))
print(len(bcn_metadata))
bcn_metadata.head()

# %%
# Map multiclass target
bcn_data = bcn_metadata.copy()
bcn_data['multiclass']=bcn_data['diagnosis'].map(class_mapping)
bcn_data['dataset']='bcn'
bcn_data['image_id']=bcn_data['isic_id'].map(lambda x: x+".jpg")
bcn_data['binary']=bcn_data['multiclass'].map(binary_mapping)
bcn_data['image_type']='dermatoscopic'
bcn_data = bcn_data[bcn_data['multiclass'].isna()==False].copy()
bcn_data = bcn_data[['image_id','binary','multiclass','dataset','image_type']]

bcn_data.to_csv('processed_data/bcn_data.csv',index=False)
processed_dfs.append(bcn_data)
print(len(bcn_data))
bcn_data.head()

# %% [markdown]
# ## DERM12345

# %%
derm12345_metadata = pd.read_csv(os.path.join(derm12345_src,'metadata.csv'))
print(len(derm12345_metadata))
derm12345_metadata.head()

# %%
derm12345_data = derm12345_metadata.copy()
derm12345_data['multiclass']=derm12345_data['diagnosis_3'].map(class_mapping)
derm12345_data['dataset']='derm12345'
derm12345_data['image_id']=derm12345_data['isic_id'].map(lambda x: x+".jpg")
derm12345_data['binary']=derm12345_data['multiclass'].map(binary_mapping)
derm12345_data['image_type']='dermatoscopic'
derm12345_data = derm12345_data[derm12345_data['multiclass'].isna()==False].copy()
derm12345_data = derm12345_data[['image_id','binary','multiclass','dataset','image_type']]

#reduce nevi amount to 1000 images
derm12345_data_subset = derm12345_data[derm12345_data['multiclass'] == "NEV"]
sampled_target_rows = derm12345_data_subset.sample(n=1000, random_state=42)
derm12345_data_others = derm12345_data[derm12345_data['multiclass'] != "NEV"]
derm12345_data = pd.concat([derm12345_data_others, sampled_target_rows], ignore_index=True)

derm12345_data.to_csv('processed_data/derm12345_data.csv',index=False)
processed_dfs.append(derm12345_data)
print(len(derm12345_data))
derm12345_data.head()

# %% [markdown]
# ## DERM7PT

# %%
derm7pt_metadata = pd.read_csv(os.path.join(derm7pt_src,'meta.csv'))
# Seperate derm and clinical image to their own row
df_derm = derm7pt_metadata.rename(columns={'derm': 'image_id'})
df_derm['image_type'] = 'dermatoscopic'
df_clinic = derm7pt_metadata.rename(columns={'clinic': 'image_id'})
df_clinic['image_type'] = 'clinical'
derm7pt_metadata = pd.concat([df_derm, df_clinic], ignore_index=True)
derm7pt_metadata

# %%
derm7pt_data = derm7pt_metadata.copy()
derm7pt_data['multiclass']=derm7pt_data['diagnosis'].map(class_mapping)
derm7pt_data.value_counts('multiclass')
derm7pt_data['dataset']='derm7pt'
derm7pt_data['binary']=derm7pt_data['multiclass'].map(binary_mapping)
derm7pt_data['image_id']=derm7pt_data['image_id'].str.split('/').str[-1]
derm7pt_data = derm7pt_data[derm7pt_data['multiclass'].isna()==False].copy()

derm7pt_data = derm7pt_data[['image_id','binary','multiclass','dataset','image_type']]
processed_dfs.append(derm7pt_data)
derm7pt_data.to_csv('processed_data/derm7pt_data.csv',index=False)
print(len(derm7pt_data))
derm7pt_data.head()

# %% [markdown]
# ## MedNode

# %%
med_node_metadata = pd.read_csv(os.path.join(med_node_src,"metadata.csv"))
print(len(med_node_metadata))
med_node_metadata.head()

# %%
# Map multiclass target
med_node_data = med_node_metadata.copy()
med_node_data['binary']=med_node_data['multiclass'].map(binary_mapping)
med_node_data['dataset']='med_node'
med_node_data['image_id']=med_node_data['image_id'].str.split('/').str[-1]
med_node_data['image_type']='clinical'
med_node_data = med_node_data[med_node_data['multiclass'].isna()==False].copy()

med_node_data= med_node_data[['image_id','binary','multiclass','dataset','image_type']]
processed_dfs.append(med_node_data)
med_node_data.to_csv('processed_data/med_node_data.csv',index=False)
print(len(med_node_data))
med_node_data.head()

# %% [markdown]
# ## SD-198

# %%
sd_198_metadata = pd.read_csv(os.path.join(sd_198_src,"metadata.csv"))
print(len(sd_198_metadata))
sd_198_metadata.head()

# %%
sd_198_data = sd_198_metadata.copy()
sd_198_data['multiclass']=sd_198_data['label'].map(class_mapping)
sd_198_data['binary']=sd_198_data['multiclass'].map(binary_mapping)
sd_198_data['dataset']='sd_198'
sd_198_data['image_type']='clinical'
sd_198_data = sd_198_data[sd_198_data['multiclass'].isna()==False].copy()

sd_198_data= sd_198_data[['image_id','binary','multiclass','dataset','image_type']]
processed_dfs.append(sd_198_data)
sd_198_data.to_csv('processed_data/sd_198_data.csv',index=False)
print(len(sd_198_data))
sd_198_data.head()

# %% [markdown]
# ## Combine Data

# %%
combined_df = pd.concat(processed_dfs, ignore_index=True)
print(len(combined_df))
combined_df.head()

# %%
#Validate images exist
image_root_dirs = {
        'ddi': os.path.join(data_dir, 'ddi/images'),
        'fp': os.path.join(data_dir, 'fritzpatrick/images/huggingface_fritzpatrick'),
        'pad': os.path.join(data_dir, 'pad-ufes-20/images'),
        'scin': os.path.join(data_dir, 'scin/images'),
        'mskcc': os.path.join(data_dir, 'MSKCC/images'),
        'argentina': os.path.join(data_dir, 'argentina/images'),
        'isic_24': os.path.join(data_dir, 'isic_24/images'),
        'ham': os.path.join(data_dir, 'ham10000/images'),
        'bcn': os.path.join(data_dir, 'bcn20000/images'),
        'derm12345': os.path.join(data_dir, 'derm12345/images'),
        'derm7pt': os.path.join(data_dir, 'derm7pt/images'),
        'med_node': os.path.join(data_dir, 'med_node/images'),
        'sd_198': os.path.join(data_dir, 'sd-198/images'),
    }
filtered_dfs = []
for dataset in image_root_dirs:
    root = image_root_dirs[dataset]
    df_subset = combined_df[combined_df['dataset'] == dataset].copy()  
    df_filtered = df_subset[df_subset['image_id'].apply(lambda x: os.path.exists(os.path.join(root, x)))]
    filtered_dfs.append(df_filtered)

# Combine all filtered subsets
combined_df_validated = pd.concat(filtered_dfs, ignore_index=True)
combined_df_validated

# %%
combined_df_validated.to_csv('processed_data/combined_data.csv',index=False)

# %%
combined_df.value_counts("dataset",dropna=False)


