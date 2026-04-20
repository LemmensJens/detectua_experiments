import pandas as pd

# placeholder for final dataset
data = {
    'corpus': [],
    'genre': [],
    'text': [],
    'binary_label': [],
    'multiclass_label': [],
}

# CLIN dataset
clin_df = pd.read_csv('clin33_shared_task_test_nl.csv')

data['corpus'].extend(['CLIN']*len(clin_df))
clin_df['genre'] = clin_df['genre'].apply(lambda x: 'Social Media' if x=='Twitter' else x)
data['genre'].extend(clin_df['genre'])
data['text'].extend(clin_df['text'])
data['binary_label'].extend(clin_df['label'])
data['multiclass_label'].extend(clin_df['label'])
# data['multiclass_label'] = data['multiclass_label'].extend(clin_df['multiclass_label'])

# CSI corpus - reviews
csi_reviews_df = pd.read_csv('csi_reviews_ai_generated.csv')

"""
First split the corpus in half (randomly). Then use half of the rows as human data and other half as AI data
"""

column_names = ['neutral_text_gemma3:12b', 'neutral_text_phi4:14b', 'neutral_text_deepseek-r1:14b', 'neutral_text_llama2:13b', 'neutral_text_qwen2.5:14b']
csi_reviews_1 = csi_reviews_df.sample(frac=0.5, random_state=42)      # random 50%

data['corpus'].extend(['CSI']*len(csi_reviews_1))
data['genre'].extend(['Reviews']*len(csi_reviews_1))
data['text'].extend(csi_reviews_1['text_human'])
data['binary_label'].extend([0]*len(csi_reviews_1))
data['multiclass_label'].extend(['human']*len(csi_reviews_1))

csi_reviews_2 = csi_reviews_df.drop(csi_reviews_1.index)

for column in column_names: # add all generated columns as AI data
    data['corpus'].extend(['CSI']*len(csi_reviews_2))
    data['genre'].extend(['Reviews']*len(csi_reviews_2))
    data['binary_label'].extend([1]*len(csi_reviews_2))
    data['text'].extend(csi_reviews_2[column])
    data['multiclass_label'].extend([column.split('_')[-1]]*len(csi_reviews_2))

# CSI corpus - essays
csi_essays_df = pd.read_csv('csi_essays_ai_generated.csv')
csi_essays_1 = csi_essays_df.sample(frac=0.5, random_state=42)      # random 50%

data['corpus'].extend(['CSI']*len(csi_essays_1))
data['genre'].extend(['Essays']*len(csi_essays_1))
data['text'].extend(csi_essays_1['text_human'])
data['binary_label'].extend([0]*len(csi_essays_1))
data['multiclass_label'].extend(['human']*len(csi_essays_1))

csi_essays_2 = csi_essays_df.drop(csi_essays_1.index)

for column in column_names:
    data['corpus'].extend(['CSI']*len(csi_essays_2))
    data['genre'].extend(['Essays']*len(csi_essays_2))
    data['binary_label'].extend([1]*len(csi_essays_2))
    data['text'].extend(csi_essays_2[column])
    data['multiclass_label'].extend([column.split('_')[-1]]*len(csi_essays_2))

# CLiPS News corpus
news_df = pd.read_csv('nieuws_ai_generated.csv')
news_1 = news_df.sample(frac=0.5, random_state=42)      # random 50%

data['corpus'].extend(['News']*len(news_1))
data['genre'].extend(['News']*len(news_1))
data['text'].extend(news_1['text_human'])
data['binary_label'].extend([0]*len(news_1))
data['multiclass_label'].extend(['human']*len(news_1))

news_2 = news_df.drop(news_1.index)

for column in column_names:
    data['corpus'].extend(['News']*len(news_2))
    data['genre'].extend(['News']*len(news_2))
    data['binary_label'].extend([1]*len(news_2))
    data['text'].extend(news_2[column])
    data['multiclass_label'].extend([column.split('_')[-1]]*len(news_2))

# Combine all
df = pd.DataFrame(data=data)
df['length'] = df['text'].apply(lambda x: len(x.split()))
df.dropna(subset=["text"], inplace=True)
df.drop_duplicates(subset=["text"], inplace=True)
df.to_csv('detectua_data_combined.csv', index=False)

df_no_social_media = df[df['genre']!='Social Media']
df_no_social_media.to_csv('detectua_data_combined_no_social_media.csv', index=False)

df_min_length = df[df['length']>=200]
df_min_length.to_csv('detectua_data_combined_no_social_media.csv', index=False)
print(df_min_length)
print(df_min_length['genre'].value_counts())