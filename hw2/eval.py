import pandas as pd
from bleu_eval import BLEU

df = pd.read_json('./output.json')

ref_df = pd.read_json('./MLDS_hw2_data/testing_public_label.json')
ref_df.rename(columns={'caption': 'ref'}, inplace=True)
df = df.merge(ref_df, on='id')

scores = []
for index, row in df.iterrows():
    score = 0
    for r in row.ref:
        score += BLEU(row.caption, r)
        print(row.caption)
        print(r)
    score /= len(row.ref)
    scores.append(score)
print(sum(scores)/len(scores))