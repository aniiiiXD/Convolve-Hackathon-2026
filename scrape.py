import requests
from bs4 import BeautifulSoup
import json
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
        # Find main content area
        main_content = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
        
        if not main_content:
            # Fallback: try to find content by common patterns
            main_content = soup.find('body')
        
        if main_content:
            # Remove navigation, headers, footers
            for tag in main_content.find_all(['nav', 'header', 'footer', 'script', 'style']):
                tag.decompose()
            
            # Get text with some structure preserved
            paragraphs = []
            for elem in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                text = elem.get_text(strip=True)
                if text:
                    paragraphs.append(text)
            
            return '\n\n'.join(paragraphs)
        
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
    
    def save_to_json(self, data: Dict, filename: str = "scraped_data.json"):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filename}")
    
    def save_to_text(self, data: Dict, filename: str = "scraped_data.txt"):
        """Save scraped content to text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Title: {data.get('title', 'N/A')}\n")
            f.write(f"URL: {data.get('url', 'N/A')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(data.get('content', 'No content'))
        print(f"Content saved to {filename}")


def main():
    # Initialize scraper
    scraper = QdrantDocScraper()
    
    # Scrape all URLs from the QD_DOCS_URLS list
    for url in QD_DOCS_URLS:
        print(f"\nScraping {url}...")
        
        try:
            data = scraper.scrape_page(url)
            
            if 'error' not in data:
                # Create a safe filename from the URL
                filename_base = url.replace('https://', '').replace('/', '_').rstrip('_')
                if not filename_base:
                    filename_base = 'index'
                
                # Save results
                json_file = f"{filename_base}.json"
                txt_file = f"{filename_base}.txt"
                
                scraper.save_to_json(data, json_file)
                scraper.save_to_text(data, txt_file)
                
                print(f"✓ Successfully scraped and saved {url}")
                print(f"   Title: {data.get('title', 'N/A')}")
                print(f"   Content length: {len(data.get('content', ''))} characters")
                print(f"   Navigation links found: {len(data.get('navigation', []))}")
            else:
                print(f"✗ Failed to scrape {url}: {data['error']}")
                
        except Exception as e:
            print(f"✗ Error processing {url}: {str(e)}")


if __name__ == "__main__":
    main()