import pandas as pd
import requests
import os
import mimetypes
from urllib.parse import urlparse

#Configuration
csv_file_path ="data/new_encoded.csv"
image_url_column = 'link'
image_name_column = 'shortcode'
output_folder = 'data/img_folder'

df = pd.read_csv(csv_file_path)

for i, row in df.iterrows():
    url = row[image_url_column]
    filename_base = str(row[image_name_column])

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        #Get File extension from content-header type
        content_type = response.headers.get('Content-Type')
        ext = mimetypes.guess_extension(content_type.split(";")[0]) if content_type else '.jpg'
        if not ext:
            ext = 'jpg' #Fallback
        
        filename = f"{filename_base}{ext}"
        image_path = os.path.join(output_folder, filename)

        with open(image_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded: {filename}")

    except Exception as e:
        print(f"Failed to download image at row {i}: {e}")