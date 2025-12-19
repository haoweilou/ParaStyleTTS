import pandas as pd
from g2p import mix_to_ipa
data = pd.read_csv("fileloader/example.csv",delimiter="\t")
data[['ipa', 'style']] = data['sentence'].apply(
    lambda x: pd.Series(mix_to_ipa(x))
)
data.to_csv("fileloader/example_processed.csv", sep="\t", index=False)