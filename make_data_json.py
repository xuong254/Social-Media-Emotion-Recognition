# make_data_json.py
import pandas as pd
import json

files = ['train_nor_811.xlsx', 'valid_nor_811.xlsx', 'test_nor_811.xlsx']
dfs = []
for f in files:
    df = pd.read_excel(f)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# lấy 2 cột chính (tùy tên cột trong file)
text_col = [c for c in df_all.columns if 'sent' in c.lower() or 'text' in c.lower()][0]
label_col = [c for c in df_all.columns if 'emotion' in c.lower() or 'label' in c.lower()][0]

df_all = df_all[[text_col, label_col]].dropna()
df_all.columns = ['Sentence','Emotion']

# lưu thành JSON để frontend hiển thị
df_all.to_json('data.json', orient='records', force_ascii=False, indent=2)
print(f"✅ Đã tạo data.json với {len(df_all)} câu.")
print(f"🔎 Nhãn duy nhất: {df_all['Emotion'].unique()}")
