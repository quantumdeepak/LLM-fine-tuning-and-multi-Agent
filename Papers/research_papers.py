import os
import time
import re
import requests
from bs4 import BeautifulSoup
import datetime
from urllib.parse import urljoin

# ============================ HYPERPARAMETERS ============================ #

START_YEAR = 2013  # Change this to the earliest year you want to download
CONFERENCES = ['CVPR', 'ICCV', 'WACV']  # Conferences to download
DOWNLOAD_PATH = "cvf_papers"  # Directory to save downloaded PDFs
SLEEP_BETWEEN_DOWNLOADS = 1  # Sleep time between downloads to avoid overloading the server
MAX_RETRIES = 3  # Maximum number of retries for failed downloads
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# ======================================================================== #

# Ensure the download directory exists
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

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
        headers = {"User-Agent": USER_AGENT}
        response = requests.head(url, headers=headers, timeout=10)
        return 200 <= response.status_code < 300
    except:
        return False

def get_paper_links(url):
    """Extract all PDF links from a conference page."""
    print(f"Fetching paper links from: {url}")
    paper_links = []
    
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for paper links using various patterns (may vary by year/conference)
        # Pattern 1: Direct PDF links
        for link in soup.find_all('a', href=lambda href: href and href.endswith('.pdf')):
            href = link.get('href')
            # Make sure we have absolute URLs
            if not href.startswith('http'):
                href = urljoin(url, href)
            paper_links.append(href)
        
        # Pattern 2: Links to paper detail pages
        for link in soup.find_all('a', href=lambda href: href and 'paper_id' in href):
            detail_url = urljoin(url, link.get('href'))
            # Get PDF link from detail page
            try:
                detail_response = requests.get(detail_url, headers=headers, timeout=10)
                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                pdf_link = detail_soup.find('a', href=lambda href: href and href.endswith('.pdf'))
                if pdf_link:
                    href = pdf_link.get('href')
                    if not href.startswith('http'):
                        href = urljoin(url, href)
                    paper_links.append(href)
                time.sleep(0.5)  # Be gentle to the server
            except Exception as e:
                print(f"Error fetching detail page {detail_url}: {e}")
        
        # Deduplicate links
        paper_links = list(set(paper_links))
        print(f"Found {len(paper_links)} paper links")
        return paper_links
        
    except Exception as e:
        print(f"Error fetching paper links from {url}: {e}")
        return []

def download_file(url, file_path):
    """Downloads a file with retry mechanism."""
    for attempt in range(MAX_RETRIES):
        try:
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)
                        # Simple progress indicator
                        if total_size > 0:
                            progress = downloaded / total_size * 100
                            print(f"\rProgress: {progress:.1f}%", end="")
            
            print()  # New line after progress indicator
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Download failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(2)  # Wait before retrying
            
    return False

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
    successful_downloads = 0
    
    for i, link in enumerate(paper_links):
        try:
            # Extract filename from the URL
            raw_filename = link.split('/')[-1]
            filename = sanitize_filename(raw_filename)
            file_path = os.path.join(year_dir, filename)
            
            # Skip download if the file already exists
            if os.path.exists(file_path):
                print(f"✔ Skipping {filename} (already downloaded)")
                successful_downloads += 1
                continue
            
            print(f"Downloading ({i+1}/{len(paper_links)}): {filename}")
            
            if download_file(link, file_path):
                print(f"✔ Downloaded {filename}")
                successful_downloads += 1
            else:
                print(f"❌ Failed to download {filename} after multiple attempts")
            
            # Sleep to avoid overwhelming the server
            time.sleep(SLEEP_BETWEEN_DOWNLOADS)
            
        except Exception as e:
            print(f"❌ Failed to process {link}: {e}")
    
    print(f"✅ Completed downloads for {conference} {year}: {successful_downloads}/{len(paper_links)} successful\n")
    return successful_downloads

# ============================ MAIN FUNCTION ============================ #

def main():
    current_year = get_current_year()
    conference_urls = get_conference_urls()
    
    total_papers = 0
    
    try:
        for conference in CONFERENCES:
            print(f"\n{'='*50}")
            print(f"Processing {conference} conferences from {START_YEAR} to {current_year}")
            print(f"{'='*50}\n")
            
            for year in range(START_YEAR, current_year + 1):
                url = conference_urls[conference][year]
                papers_count = download_conference_papers(conference, year, url)
                total_papers += papers_count
                
        print(f"\n✅ All conference papers download attempts completed! Total papers: {total_papers}")
        
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user. Exiting gracefully...")
        
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        
if __name__ == "__main__":
    main()