# Cliffjumper
Neural Search.

`tatoeba.en-zh`: From  https://drive.google.com/open?id=1AIUAbU-GkPFN0nahRHaK8nV7gtLk68fG

<!--
<a href="https://www.codecogs.com/eqnedit.php?latex=\LARGE&space;x&space;=&space;a_0&space;&plus;&space;\frac{1}{a_1&space;&plus;&space;\frac{1}{a_2&space;&plus;&space;\frac{1}{a_3&space;&plus;&space;a_4}}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\LARGE&space;x&space;=&space;a_0&space;&plus;&space;\frac{1}{a_1&space;&plus;&space;\frac{1}{a_2&space;&plus;&space;\frac{1}{a_3&space;&plus;&space;a_4}}}" title="\LARGE x = a_0 + \frac{1}{a_1 + \frac{1}{a_2 + \frac{1}{a_3 + a_4}}}" /></a>
-->

Indexing
====

```python

from tqdm import tqdm
import numpy as np
import nmslib

import torch
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification

from transformers import BertTokenizer, BertModel

#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')

def vectorize(text):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    return model(input_ids)[1].squeeze().detach().numpy()
    
sentences = []
with open('tatoeba.en-zh') as fin:
    for line in fin:
        if line.strip():
            en, zh = line.strip().split('\t')
            sentences.append(en)
            sentences.append(zh)
            
sentences = list(set(sentences)) # Unique list.

# Converts sentences to arrays of floats.
vectorized_sents = [vectorize(s) for s in tqdm(sentences)]

# Concatenate the arrays.
data = np.vstack(vectorized_sents)

# Create the index
index = nmslib.init(method='hnsw', space='cosinesimil')
# Add data to index.
index.addDataPointBatch(data)
# The actual indexing.
index.createIndex({'post': 2}, print_progress=True)
```

Querying
====

```python
# When using the index.

# Convert single string to array of floats.
query = vectorize("how fast is the car?")

ids, distances = index.knnQuery(query, k=10) # k=10 means top-10 results
# Results.
for i in ids:
    print(sentences[i])
```
