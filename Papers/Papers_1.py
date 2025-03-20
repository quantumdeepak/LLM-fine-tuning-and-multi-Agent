import os
import time
import re
import requests
from bs4 import BeautifulSoup
import datetime
from urllib.parse import urljoin
import concurrent.futures
from tqdm import tqdm  # For better progress bars

# ============================ HYPERPARAMETERS ============================ #

START_YEAR = 2013  # Change this to the earliest year you want to download
CONFERENCES = ['CVPR', 'ICCV', 'WACV']  # Conferences to download
DOWNLOAD_PATH = "cvf_papers"  # Directory to save downloaded PDFs
MAX_RETRIES = 3  # Maximum number of retries for failed downloads
MAX_WORKERS = 32  # Maximum number of parallel downloads
SESSION_TIMEOUT = 30  # Timeout for HTTP requests in seconds
DETAIL_PAGE_WORKERS = 16  # Workers for fetching detail pages
DOWNLOAD_WORKERS = 32  # Workers for downloading files
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# ======================================================================== #

# Ensure the download directory exists
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

# Create a session object to reuse connections
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
session.mount('https://', adapter)
session.mount('http://', adapter)

def get_current_year():
    """Returns the current year."""
    return datetime.datetime.now().year

def get_conference_urls():
    """Generates URLs for different conferences and years."""
    conference_urls = {}
    current_year = get_current_year()
    
    for conference in CONFERENCES:
        conference_urls[conference] = {}
        for year in range(START_YEAR, current_year + 1):
            # Basic URL patterns
            if year < 2018:
                # For older conferences
                url = f"https://openaccess.thecvf.com/{conference}{year}"
            else:
                # For newer conferences
                url = f"https://openaccess.thecvf.com/{conference}{year}?day=all"
            
            conference_urls[conference][year] = url
    
    return conference_urls

def check_url_exists(url):
    """Check if the URL exists and returns a valid page."""
    try:
        response = session.head(url, timeout=SESSION_TIMEOUT)
        return 200 <= response.status_code < 300
    except:
        return False

def get_pdf_from_detail_page(detail_url, base_url):
    """Extract PDF link from a paper detail page."""
    try:
        response = session.get(detail_url, timeout=SESSION_TIMEOUT)
        detail_soup = BeautifulSoup(response.text, 'html.parser')
        pdf_link = detail_soup.find('a', href=lambda href: href and href.endswith('.pdf'))
        if pdf_link:
            href = pdf_link.get('href')
            if not href.startswith('http'):
                href = urljoin(base_url, href)
            return href
        return None
    except Exception:
        return None

def get_paper_links(url):
    """Extract all PDF links from a conference page."""
    print(f"Fetching paper links from: {url}")
    paper_links = set()  # Use a set to avoid duplicates
    detail_urls = []
    
    try:
        response = session.get(url, timeout=SESSION_TIMEOUT)
        if not response.ok:
            print(f"Failed to access {url}, status code: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Direct PDF links
        for link in soup.find_all('a', href=lambda href: href and href.endswith('.pdf')):
            href = link.get('href')
            # Make sure we have absolute URLs
            if not href.startswith('http'):
                href = urljoin(url, href)
            paper_links.add(href)
        
        # Collect detail page URLs
        for link in soup.find_all('a', href=lambda href: href and ('paper_id' in href or 'papers' in href)):
            detail_url = urljoin(url, link.get('href'))
            detail_urls.append(detail_url)
        
        # Process detail pages in parallel
        if detail_urls:
            print(f"Found {len(detail_urls)} detail pages to process")
            with concurrent.futures.ThreadPoolExecutor(max_workers=DETAIL_PAGE_WORKERS) as executor:
                futures = {executor.submit(get_pdf_from_detail_page, detail_url, url): detail_url for detail_url in detail_urls}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing detail pages"):
                    pdf_url = future.result()
                    if pdf_url:
                        paper_links.add(pdf_url)
        
        result = list(paper_links)
        print(f"Found {len(result)} unique paper links")
        return result
        
    except Exception as e:
        print(f"Error fetching paper links from {url}: {e}")
        return []

def download_file(url, file_path):
    """Downloads a file with retry mechanism."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, stream=True, timeout=SESSION_TIMEOUT)
            response.raise_for_status()
            
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            
            return True
            
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)  # Wait before retrying
            else:
                print(f"Download failed after {MAX_RETRIES} attempts: {url}")
            
    return False

def download_paper(args):
    """Function for parallel downloading of a single paper."""
    link, file_path, i, total = args
    
    try:
        # Skip download if the file already exists
        if os.path.exists(file_path):
            return (True, file_path, "already exists")
        
        success = download_file(link, file_path)
        if success:
            return (True, file_path, "downloaded")
        else:
            return (False, file_path, "failed")
        
    except Exception as e:
        return (False, file_path, str(e))

def sanitize_filename(filename):
    """Sanitize filename to remove invalid characters."""
    # Remove invalid characters for filenames
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def download_conference_papers(conference, year, url):
    """Downloads papers for a specific conference and year."""
    if not check_url_exists(url):
        print(f"⚠️ URL not available: {url}")
        return 0
    
    paper_links = get_paper_links(url)
    
    if not paper_links:
        print(f"No papers found for {conference} {year}.")
        return 0
    
    conf_dir = os.path.join(DOWNLOAD_PATH, conference)
    year_dir = os.path.join(conf_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    
    print(f"Downloading {len(paper_links)} papers for {conference} {year}...")
    
    # Prepare arguments for parallel downloads
    download_args = []
    for i, link in enumerate(paper_links):
        # Extract filename from the URL
        raw_filename = link.split('/')[-1]
        filename = sanitize_filename(raw_filename)
        file_path = os.path.join(year_dir, filename)
        download_args.append((link, file_path, i+1, len(paper_links)))
    
    # Download files in parallel
    successful_downloads = 0
    failed_downloads = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(download_paper, arg) for arg in download_args]
        
        # Show progress with tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{conference} {year}"):
            success, file_path, status = future.result()
            if success:
                successful_downloads += 1
            else:
                failed_downloads += 1
    
    print(f"✅ Completed {conference} {year}: {successful_downloads} successful, {failed_downloads} failed\n")
    return successful_downloads

# ============================ MAIN FUNCTION ============================ #

def main():
    current_year = get_current_year()
    conference_urls = get_conference_urls()
    
    total_papers = 0
    start_time = time.time()
    
    try:
        # Process each conference in a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(CONFERENCES)) as executor:
            # Submit one task per conference-year combination
            futures = []
            for conference in CONFERENCES:
                print(f"\nProcessing {conference} conferences from {START_YEAR} to {current_year}")
                
                for year in range(START_YEAR, current_year + 1):
                    url = conference_urls[conference][year]
                    future = executor.submit(download_conference_papers, conference, year, url)
                    futures.append((future, conference, year))
            
            # Process results as they complete
            for future, conference, year in futures:
                papers_count = future.result()
                total_papers += papers_count
                
        elapsed_time = time.time() - start_time
        print(f"\n✅ All downloads completed in {elapsed_time:.2f} seconds!")
        print(f"Total papers downloaded: {total_papers}")
        
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user. Exiting gracefully...")
        
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        
if __name__ == "__main__":
    main()