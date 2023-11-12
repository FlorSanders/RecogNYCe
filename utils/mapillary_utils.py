import os
import numpy as np
import json
from dotenv import load_dotenv
from functools import lru_cache
import aiohttp
import aiofiles
import asyncio
import pandas as pd

# Load token from .env (security yay)
# See: https://help.mapillary.com/hc/en-us/articles/360010234680-Accessing-imagery-and-data-through-the-Mapillary-API
load_dotenv()
token = os.getenv("MAPILLARY_TOKEN")
assert token, "Please create a .env file and add your mapilarry token under the MAPILLARY_TOKEN key"


@lru_cache(1) # Add 1-lru cache to avoid computing this over and over again
def radius_to_angle(r):
    """
    Convert a given radius to a longitude/latitude angle that can be used to determine the bounding box
    ---
    Args:
    - r: Radius
    
    Returns:
    - angle: Longitude / latitude angle
    """
    R = 6371e3
    theta = np.arcsin(r / R)
    angle = theta / np.pi * 180
    return angle

async def download_images_async(coords, neighborhood, data_dir="./data/mapillary/", radius=30, resolution=2048, batch_size=25, neighborhood_is_borough=True, verbose=True):
    """
    Asynchronously download images from mapilarry
    ---
    Args:
    - coords: list of coordinates (longitude, latitude) to look for images at
    - neighborhood: what neighborhood the coordinate is part of
    - data_dir: Directory to store the data
    - radius: Radius in meters around the coordinate to look for images
    - resolution: Thumbnail resolution of the images to download
    - batch_size: How many images to download in parallel
    - neighborhood_is_borough: Whether the neighborhood label should be saved under "borough" or "neighborhood"

    Returns:
    - Dataset: Pandas dataframe containing the downloaded images
    """
    results = []
    n_batches = int(np.ceil(len(coords) / batch_size))
    for batch in range(n_batches):
        batch_coords = coords[batch*batch_size:(batch+1)*batch_size]
        batch_results = await download_image_batch_async(batch_coords, neighborhood, data_dir, radius, resolution, verbose=verbose)
        results.extend([result for result in batch_results if result is not None])
    

    results_df = pd.DataFrame(results, columns=["id", "longitude", "latitude", "path", "url"])
    neighborhood_col = "borough" if neighborhood_is_borough else "neighborhood"
    results_df[neighborhood_col] = neighborhood

    return results_df


async def download_image_batch_async(coords, neighborhood, data_dir="./data/mapillary/", radius=30, resolution=2048, verbose=True):
    # Make sure save path exists
    neighborhood_path = os.path.join(data_dir, neighborhood)
    os.makedirs(neighborhood_path, exist_ok=True)

    # Download images at coords
    async with aiohttp.ClientSession() as session:
        tasks = []
        for coord in coords:
            task = download_image_async(session, coord, neighborhood_path, radius=radius, resolution=resolution, verbose=verbose)
            tasks.append(asyncio.ensure_future(task))
        results = await asyncio.gather(*tasks)
        return results

async def download_image_async(session, coord, neighborhood_path, radius=30, resolution=2048, verbose=True):
    # 1. Look up images within radius
    # Construct url
    angle = radius_to_angle(radius)
    url_base = f"https://graph.mapillary.com/images?access_token={token}&is_pano=false&fields=id,thumb_{resolution}_url,geometry&limit=1&bbox="
    url = f"{url_base}{coord[0]-angle:.5f},{coord[1]-angle:.5f},{coord[0]+angle:.5f},{coord[1]+angle:.5f}"

    # Fetch url
    try:
        async with session.get(url) as res:
            # Parse body
            body = await res.json()
            data = body.get("data", [])
            if not len(data):
                if verbose:
                    print(f"WARNING: No images found for coord = {coord}")
                return None
            
            # Destructure body
            img = data[0]
            img_id = img["id"]
            img_url = img[f"thumb_{resolution}_url"]
            img_coords = img["geometry"]["coordinates"]

        async with session.get(img_url) as img_res:
            if img_res.status != 200:
                if verbose:
                    print(f"WARNING: Unable to download image from url {img_url}")
                return None
            img_path = os.path.join(neighborhood_path, f"{img_id}.jpg") 
            f = await aiofiles.open(img_path, mode='wb')
            await f.write(await img_res.read())
            await f.close()            
            return img_id, img_coords[0], img_coords[1], img_path, img_url
    except Exception as e:
        if verbose:
            print(f"WARNING: Unable to get image for coord = {coord}, because {e.__class__}")
            print(e)
        return None

