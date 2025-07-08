# %%
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# %%
# Read Processed data
processed_df = pd.read_csv("processed_data/combined_data.csv")
processed_df

# %%
# Validate image existance
data_dir = '../../data/'
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
    df_subset = processed_df[processed_df['dataset'] == dataset]  
    df_filtered = df_subset[df_subset['image_id'].apply(lambda x: os.path.exists(os.path.join(root, x)))]
    filtered_dfs.append(df_filtered)

# Combine all filtered subsets
combined_df_validated = pd.concat(filtered_dfs, ignore_index=True)
combined_df_validated

# %% [markdown]
# ## Skin Tone

# %%
save_dir = "skin_tone"
os.makedirs(save_dir, exist_ok=True)
#Get skin tone subset
skin_tone_df = combined_df_validated.dropna(subset=['skin_tone']).copy()
#Add stratification column
skin_tone_df['stratify_col'] = skin_tone_df['skin_tone'].astype(str) + '_' + skin_tone_df['image_type'].astype(str)
skin_tone_df.to_csv(os.path.join(save_dir, "skin_tone_data.csv"), index=False)
skin_tone_df


# %%
#Create Train Test Splits
train_df, test_df = train_test_split(
    skin_tone_df,
    test_size=0.2,
    stratify=skin_tone_df['stratify_col'],
    random_state=42
)

# Save splits
train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

#save derma and clinical subsets
train_derma = train_df[train_df['image_type']=='dermatoscopic']
train_derma.to_csv(os.path.join(save_dir, "train_derma.csv"), index=False)
test_derma = test_df[test_df['image_type']=='dermatoscopic']
test_derma.to_csv(os.path.join(save_dir, "test_derma.csv"), index=False)

train_clinical = train_df[train_df['image_type']=='clinical']
train_clinical.to_csv(os.path.join(save_dir, "train_clinical.csv"), index=False)
test_clinical = test_df[test_df['image_type']=='clinical']
test_clinical.to_csv(os.path.join(save_dir, "test_clinical.csv"), index=False)

print(len(train_df), len(train_derma), len(train_clinical))
print(len(test_df), len(test_derma), len(test_clinical))

# %% [markdown]
# ## Multiclass

# %%
#read data with pseudolabels
processed_df = pd.read_csv("processed_data/combined_data_with_pseudo.csv")
processed_df

# %%
save_dir = "multiclass"
os.makedirs(save_dir, exist_ok=True)
#Get skin tone subset
multiclass_df = processed_df.dropna(subset=['multiclass']).copy()
#Add stratification column
multiclass_df['stratify_col'] = multiclass_df['combined_skin_tone'].astype(str) + '_' + multiclass_df['multiclass'].astype(str)
multiclass_df.to_csv(os.path.join(save_dir, "multiclass_data.csv"), index=False)
multiclass_df

# %%
#Create Train Test Splits
train_df, test_df = train_test_split(
    multiclass_df,
    test_size=0.2,
    stratify=multiclass_df['stratify_col'],
    random_state=42
)

# Save splits
train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

#save derma and clinical subsets
train_derma = train_df[train_df['image_type']=='dermatoscopic']
train_derma.to_csv(os.path.join(save_dir, "train_derma.csv"), index=False)
test_derma = test_df[test_df['image_type']=='dermatoscopic']
test_derma.to_csv(os.path.join(save_dir, "test_derma.csv"), index=False)

train_clinical = train_df[train_df['image_type']=='clinical']
train_clinical.to_csv(os.path.join(save_dir, "train_clinical.csv"), index=False)
test_clinical = test_df[test_df['image_type']=='clinical']
test_clinical.to_csv(os.path.join(save_dir, "test_clinical.csv"), index=False)

print(len(train_df), len(train_derma), len(train_clinical))
print(len(test_df), len(test_derma), len(test_clinical))


