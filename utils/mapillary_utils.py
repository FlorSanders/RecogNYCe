import mapillary.interface as mly
import os
import requests
import numpy as np
from dotenv import load_dotenv

# Load token from .env (security yay)
load_dotenv()
token = os.getenv("MAPILLARY_TOKEN")
assert token, "Please create a .env file and add your mapilarry token under the MAPILLARY_TOKEN key"
mly.set_access_token(token)

# Download images from MLY
# TODO: reimplement with aiohttp for serious speedups!
# https://stackoverflow.com/questions/57126286/fastest-parallel-requests-in-python
def download_images(coord, save_path, radius=30, n_images=10, resolution=2048):
    # Make sure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Find images close to the coordinate
    result = mly.get_image_close_to(longitude=coord[0], latitude=coord[1], radius=radius)

    if not result:
        print(f"WARNING: Could not find image near coordinate {coord}")
        return

    # Load images
    images = np.array(result.to_dict()["features"])
    n_images = min(n_images, len(images))

    # Randomly select n_images images
    choice_indices = np.random.choice(len(images), size=n_images, replace=False)
    choices = images[choice_indices]
    if n_images == 1:
        choices = [choices] # make sure it is a list of choices

    # Download image for each choice
    for choice in choices:
        # Get img url
        choice_id = choice["properties"]["id"]
        url = mly.image_thumbnail(image_id=choice_id, resolution=resolution)

        # Download image
        img_result = requests.get(url)

        if img_result.status_code != 200:
            print(f"WARNING: img {choice_id} could not be downloaded")
            continue

        # Save image to file
        with open(os.path.join(save_path, f"img_{choice_id}.jpg"), "wb") as img_file:
            for chunk in img_result:
                img_file.write(chunk)