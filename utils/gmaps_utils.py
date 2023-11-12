from dotenv import load_dotenv
import numpy as np
import json
import aiohttp
import aiofiles
import asyncio
import pandas as pd
import os

# Load token from .env
# See: https://developers.google.com/maps/documentation/streetview/get-api-key
load_dotenv()
token = os.getenv("GMAPS_TOKEN")
assert token, "Please create a .env file and add your google maps token under the GMAPS_TOKEN key"

async def download_images_async(coords, neighborhood, data_dir="./data/gmaps", radius=30, resolution=640, batch_size=25, neighborhood_is_borough=True, verbose=True):
    # Download results in batches
    results = []
    n_batches = int(np.ceil(len(coords) / batch_size))
    for batch in range(n_batches):
        batch_coords = coords[batch*batch_size:(batch+1)*batch_size]
        batch_results = await download_image_batch_async(batch_coords, neighborhood, data_dir, radius=radius, resolution=resolution, verbose=verbose)
        results.extend([result for result in batch_results if result is not None])
    
    # Convert to dataframe
    results_df = pd.DataFrame(results, columns=["id", "longitude", "latitude", "path", "url"])
    neighborhood_col = "borough" if neighborhood_is_borough else "neighborhood"
    results_df[neighborhood_col] = neighborhood

    return results_df


async def download_image_batch_async(coords, neighborhood, data_dir="./data/gmaps/", radius=30, resolution=640, verbose=True):
    # Make sure save path exists
    neighborhood_path = os.path.join(data_dir, neighborhood)
    os.makedirs(neighborhood_path, exist_ok=True)

    # Download images at coords
    async with aiohttp.ClientSession() as session:
        tasks = []
        for coord in coords:
            task = download_image_async(session, coord, neighborhood_path, radius=radius, resolution=resolution, verbose=verbose)
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results

async def download_image_async(session, coord, neighborhood_path, radius=30, resolution=640, verbose=True):
    # Build url
    heading = np.random.randint(360)
    img_url = f"https://maps.googleapis.com/maps/api/streetview?location={coord[1]:.5f},{coord[0]:.5f}&heading={heading}&radius={radius}&size={resolution}x{resolution}&return_error_code=true&key={token}"
    
    # Fetch url
    try:
        async with session.get(img_url) as img_res:
            if img_res.status != 200:
                if verbose:
                    print(f"WARNING: Unable to download image from url {img_url}")
                return None
            img_id = np.random.randint(int(1e15))
            img_path = os.path.join(neighborhood_path, f"{img_id}.jpg")
            f = await aiofiles.open(img_path, mode="wb")
            await f.write(await img_res.read())
            await f.close()
            return img_id, coord[0] , coord[1], img_path, img_url
    except Exception as e:
        if verbose:
            print(f"WARNING: Unable to get image for coord = {coord}, because {e.__class__}")
            print(e)

if __name__ == "__main__":
    session = None
    coord = [-74.119584, 40.565845]
    neighborhood_path = None
    asyncio.run(download_image_async(session, coord, neighborhood_path))