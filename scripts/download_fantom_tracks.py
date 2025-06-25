import concurrent.futures
import os
from urllib.parse import urljoin

import requests
import tqdm
from bs4 import BeautifulSoup


def get_files_in_directory(base_url, data_path):
    """
    Scrape the FANTOM directory to get all available files

    Parameters:
    base_url (str): Base URL of FANTOM
    data_path (str): Path to the specific data directory

    Returns:
    list: List of file URLs
    """
    url = urljoin(base_url, data_path)
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Parse HTML to find all links
        soup = BeautifulSoup(response.text, "html.parser")
        files = []

        for link in soup.find_all("a"):
            href = link.get("href")
            # Look for bigwig files
            if href and href.endswith(".bw"):
                file_url = urljoin(url, href)
                files.append(file_url)

        return files
    except Exception as e:
        print(f"Error accessing directory {url}: {str(e)}")
        return []


def download_file(url, output_dir):
    """
    Download a single file from FANTOM

    Parameters:
    url (str): URL of the file
    output_dir (str): Directory to save the downloaded file

    Returns:
    str: Path to downloaded file or None if failed
    """
    try:
        filename = os.path.basename(url)
        output_path = os.path.join(output_dir, filename)

        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"File already exists: {filename}")
            return output_path

        print(f"Downloading: {filename}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get file size for progress bar
        file_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm.tqdm(total=file_size, unit="iB", unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    pbar.update(size)
        return output_path
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None


def download_fantom_cage(data_type="hCAGE", category="tissue", output_dir="~"):
    """
    Download CAGE files from FANTOM

    Parameters:
    data_type (str): Type of CAGE data ('hCAGE', 'CAGEScan', or 'LQhCAGE')
    category (str): Data category ('cell_line', 'primary_cell', 'tissue', or 'timecourse')
    output_dir (str): Directory to save downloaded files

    Returns:
    list: Paths to successfully downloaded files
    """
    base_url = "https://fantom.gsc.riken.jp/5/datahub/hg38/tpm/"

    # Construct the correct path based on category and data type
    data_path = f"human.{category}.{data_type}/"

    os.makedirs(output_dir, exist_ok=True)

    # Get all files in the directory
    urls = get_files_in_directory(base_url, data_path)

    if not urls:
        print(f"No files found in {data_path}")
        return []

    print(f"Found {len(urls)} files to download")

    # Download files in parallel
    downloaded_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(download_file, url, output_dir): url for url in urls}

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                file_path = future.result()
                if file_path:
                    downloaded_files.append(file_path)
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")

    return downloaded_files


# Example usage:
if __name__ == "__main__":
    output_dir = "data/tracks/fantom/"
    files = download_fantom_cage(
        output_dir=output_dir
        data_type="hCAGE",  # Options: 'hCAGE', 'CAGEScan', 'LQhCAGE'
        category="tissue",  # Options: 'cell_line', 'primary_cell', 'tissue', 'timecourse'
    )
    print(f"Successfully downloaded files: {files}")
