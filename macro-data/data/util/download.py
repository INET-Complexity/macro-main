import os
import shutil
import requests

from tqdm import tqdm
from pathlib import Path


def download_data(
    raw_data_path: Path,
    drive_filename: str = "1GWl07o4MRllzF3hoBGF1fKHHlH8OoTt6",
    chunk_size: int = 1024,
) -> None:
    # Download the complete dataset
    session = requests.Session()
    params = {"id": drive_filename, "confirm": 1}
    response = session.get(
        "https://docs.google.com/uc?export=download",
        params=params,
        stream=True,
    )
    total = int(response.headers.get("content-length", 0))
    with open(raw_data_path / "drive_data.zip", "wb") as f:
        for chunk in tqdm(
            response.iter_content(chunk_size),
            desc="Downloading raw data...",
            total=int(total / chunk_size),
            unit_divisor=1024,
        ):
            if chunk:
                f.write(chunk)

    # Unzip it and delete the zip file
    shutil.unpack_archive(str(raw_data_path / "drive_data.zip"), raw_data_path, format="zip")

    # Delete the zip file
    os.remove(raw_data_path / "drive_data.zip")
