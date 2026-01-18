import requests
from bs4 import BeautifulSoup
import json
import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin

# List of Qdrant documentation URLs to scrape
QD_DOCS_URLS = [
    "https://qdrant.tech/documentation/",
    "https://qdrant.tech/documentation/overview/",
    "https://qdrant.tech/documentation/quick-start/",
    "https://qdrant.tech/documentation/interfaces/",
    "https://qdrant.tech/documentation/web-ui/",
    "https://qdrant.tech/documentation/concepts/collections/",
    "https://qdrant.tech/documentation/concepts/payload/",
    "https://qdrant.tech/documentation/concepts/search/",
    "https://qdrant.tech/documentation/concepts/explore/",
    "https://qdrant.tech/documentation/concepts/hybrid-queries/",
    "https://qdrant.tech/documentation/concepts/filtering/",
    "https://qdrant.tech/documentation/concepts/inference/",
    "https://qdrant.tech/documentation/concepts/optimizer/",
    "https://qdrant.tech/documentation/concepts/points/",
    "https://qdrant.tech/documentation/concepts/storage/",
    "https://qdrant.tech/documentation/concepts/indexing/",
    "https://qdrant.tech/documentation/concepts/snapshots/",
    "https://qdrant.tech/documentation/guides/quantization/",
    "https://qdrant.tech/documentation/guides/installation/",
    "https://qdrant.tech/documentation/guides/administration/",
    "https://qdrant.tech/documentation/guides/running-with-gpu/",
    "https://qdrant.tech/documentation/guides/capacity-planning/",
    "https://qdrant.tech/documentation/guides/optimize/",
    "https://qdrant.tech/documentation/guides/multitenancy/",
    "https://qdrant.tech/documentation/guides/distributed_deployment/",
    "https://qdrant.tech/documentation/guides/text-search/",
    "https://qdrant.tech/documentation/guides/monitoring/",
    "https://qdrant.tech/documentation/guides/configuration/",
    "https://qdrant.tech/documentation/guides/security/",
    "https://qdrant.tech/documentation/guides/usage-statistics/",
    "https://qdrant.tech/documentation/guides/common-errors/",
    "https://qdrant.tech/documentation/fastembed/",
    "https://qdrant.tech/documentation/fastembed/fastembed-quickstart/",
    "https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/",
    "https://qdrant.tech/documentation/fastembed/fastembed-minicoil/",
    "https://qdrant.tech/documentation/fastembed/fastembed-splade/",
    "https://qdrant.tech/documentation/fastembed/fastembed-colbert/",
    "https://qdrant.tech/documentation/fastembed/fastembed-rerankers/",
    "https://qdrant.tech/documentation/fastembed/fastembed-postprocessing/"    
]

class QdrantDocScraper:
    def __init__(self, base_url: str = "https://qdrant.tech"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_page(self, url: str) -> Dict:
        """
        Scrape a single documentation page and extract structured content
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content
            data = {
                'url': url,
                'title': self._extract_title(soup),
                'content': self._extract_main_content(soup),
                'navigation': self._extract_navigation(soup),
                'metadata': self._extract_metadata(soup)
            }
            
            return data
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {'url': url, 'error': str(e)}
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('h1')
        return title_tag.get_text(strip=True) if title_tag else "No Title"
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from the page"""
        # Try multiple strategies to find main content
        main_content = None
        
        # Strategy 1: Look for article or main tags
        main_content = soup.find('article') or soup.find('main')
        
        # Strategy 2: Look for common content class names
        if not main_content:
            for class_name in ['content', 'documentation', 'doc-content', 'markdown', 'prose']:
                main_content = soup.find('div', class_=lambda x: x and class_name in x.lower())
                if main_content:
                    break
        
        # Strategy 3: Find the largest div with text content
        if not main_content:
            all_divs = soup.find_all('div')
            max_text_len = 0
            for div in all_divs:
                text_len = len(div.get_text(strip=True))
                if text_len > max_text_len:
                    max_text_len = text_len
                    main_content = div
        
        if main_content:
            # Clone to avoid modifying original
            content_copy = BeautifulSoup(str(main_content), 'html.parser')
            
            # Remove unwanted elements
            for tag in content_copy.find_all(['nav', 'header', 'footer', 'script', 'style', 'aside']):
                tag.decompose()
            
            # Also remove elements with specific classes that are typically navigation
            for tag in content_copy.find_all(class_=['sidebar', 'navigation', 'nav', 'menu', 'toc']):
                tag.decompose()
            
            # Get full text with better formatting
            text = content_copy.get_text(separator='\n', strip=True)
            
            # Clean up extra whitespace while preserving structure
            lines = [line.strip() for line in text.split('\n')]
            lines = [line for line in lines if line]  # Remove empty lines
            
            return '\n\n'.join(lines)
        
        return ""
    
    def _extract_navigation(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract navigation links"""
        nav_links = []
        
        # Find navigation sections
        nav_sections = soup.find_all(['nav', 'aside'])
        
        for nav in nav_sections:
            for link in nav.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                if text and href:
                    full_url = urljoin(self.base_url, href)
                    nav_links.append({
                        'text': text,
                        'url': full_url
                    })
        
        return nav_links
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata from the page"""
        metadata = {}
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            if tag.get('name'):
                metadata[tag['name']] = tag.get('content', '')
            elif tag.get('property'):
                metadata[tag['property']] = tag.get('content', '')
        
        return metadata


def save_all_to_single_file(all_data: List[Dict], txt_filename: str = "qdrant_documentation_complete.txt", json_filename: str = "qdrant_documentation.json"):
    """Save all scraped content to a single text file and a JSON file"""
    
    # Save to text file
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("QDRANT DOCUMENTATION - COMPLETE COLLECTION\n")
        f.write(f"Scraped on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total pages: {len(all_data)}\n")
        f.write("="*80 + "\n\n")
        
        for idx, page_data in enumerate(all_data, 1):
            f.write("\n" + "="*80 + "\n")
            f.write(f"PAGE {idx}/{len(all_data)}\n")
            f.write("="*80 + "\n")
            f.write(f"Title: {page_data.get('title', 'N/A')}\n")
            f.write(f"URL: {page_data.get('url', 'N/A')}\n")
            f.write("-"*80 + "\n\n")
            f.write(page_data.get('content', 'No content available'))
            f.write("\n\n")
    
    print(f"\n✓ All content saved to: {txt_filename}")
    
    # Save to JSON file
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'scraped_at': datetime.datetime.now().isoformat(),
            'total_pages': len(all_data),
            'pages': all_data
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON data saved to: {json_filename}")


def main():
    # Initialize scraper
    scraper = QdrantDocScraper()
    all_data = []
    success_count = 0
    
    print(f"Starting to scrape {len(QD_DOCS_URLS)} pages...")
    print("="*80)
    
    # Scrape all URLs from the QD_DOCS_URLS list
    for idx, url in enumerate(QD_DOCS_URLS, 1):
        print(f"\n[{idx}/{len(QD_DOCS_URLS)}] Scraping {url}...")
        
        try:
            data = scraper.scrape_page(url)
            
            if 'error' not in data:
                all_data.append({
                    'url': url,
                    'title': data.get('title', ''),
                    'content': data.get('content', ''),
                    'navigation': data.get('navigation', []),
                    'metadata': data.get('metadata', {})
                })
                success_count += 1
                print(f"✓ Successfully scraped: {data.get('title', url)}")
                print(f"  Content length: {len(data.get('content', ''))} characters")
            else:
                print(f"✗ Failed to scrape: {data['error']}")
                
        except Exception as e:
            print(f"✗ Error processing: {str(e)}")
    
    # Save all data to files
    print("\n" + "="*80)
    print("SCRAPING COMPLETE")
    print("="*80)
    
    if all_data:
        save_all_to_single_file(all_data)
        print(f"\nSuccessfully scraped: {success_count}/{len(QD_DOCS_URLS)} pages")
    else:
        print("\nNo data was scraped successfully.")


if __name__ == "__main__":
    main()