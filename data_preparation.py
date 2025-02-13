import numpy as np
import pandas as pd
import requests
import time
from pdfminer.high_level import extract_text

'''
Link to source of dataset: 
    pip install pdfminer.six
    #* REF: https://pdfminersix.readthedocs.io/en/latest/

    https://www.kaggle.com/datasets/Cornell-University/arxiv
    https://huggingface.co/datasets/arxiv-community/arxiv_dataset
'''

# Load the dataset
store = []
chunk_size = 500000
for chunk in pd.read_csv('arxiv-metadata-oai-snapshot.json', lines = True, chunksize = chunk_size):
    store.append(chunk)

data = pd.concat(store, ignore_index = True)
data['update_date'] = pd.to_datetime(data['update_date'])

# Subset for general economics published 2020 and later
GN = data[data['categories'].str.startswith('econ.GN')]
GN = GN[GN['update_date'].dt.year == 2020]
print(f'Number of papers in general economics published 2020 and later: {GN.shape[0]}')

# Download the pdf files
base_url = 'https://arxiv.org/pdf/'
def generate_pdf_link(row):
    return f"{base_url}/{row['id']}.pdf"

GN['pdf_link'] = GN.apply(generate_pdf_link, axis = 1)

def download_pdf(pdf_link, save_path):

    """ Download the pdf file
    """

    response = requests.get(pdf_link)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f'Failed to download {pdf_link}')

# Download the pdf files
save_dir = '/Users/ikedijacobs/Downloads/arxiv pdfs/' # modify the save_dir to your preferred directory

for index, row in GN.iterrows():
    pdf_link = row['pdf_link']
    save_path = f"{save_dir}{row['id']}.pdf"
    download_pdf(pdf_link, save_path)
    time.sleep(1)

def extract_text_from_pdf(pdf_path):

    """
    Extract the text from pdf file
    """

    return extract_text(pdf_path)

# Insert the text into the dataframe
body_texts = []

save_dir = '/Users/ikedijacobs/Downloads/arxiv pdfs/'
for i in GN['id']:
    pdf_path = f"{save_dir}{i}.pdf"
    body_text = extract_text_from_pdf(pdf_path)
    body_texts.append(body_text)

# Insert the body_text into the dataframe
GN['body_text'] = body_texts

# Save the dataframe
GN.to_csv('GN.csv', index = False)
GN.to_json('GN.json', orient = 'records', lines = True)