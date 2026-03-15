import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Scraper:
    def __init__(self):
        self.base_urls = [
            "https://gate.ahram.org.eg/",
            "https://m.gomhuriaonline.com/",
            "https://www.azhar.eg/splash.html",
            "https://www.dar-alifta.org/home.html",
            "https://www.islamweb.net/ar/",
            "https://www.egyptair.com/en/pages/HomePage.aspx",
            "https://study-in-egypt.gov.eg/",
            "https://www.egypttoday.com/",
            "https://grandegyptianmuseum.org/"
        ]
        self.output_file = "Data Processing Pipeline/scraped_data.json"

    def _get_article_links(self, base_url, max_links=500):
        print(f"Fetching page: {base_url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            r = requests.get(base_url, headers=headers, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"Error fetching homepage: {e}")
            return []
            
        soup = BeautifulSoup(r.text, "html.parser")
        articles = []
        seen_urls = set()

        base_netloc = urlparse(base_url).netloc
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(base_url, href)
            link_netloc = urlparse(full_url).netloc

            if link_netloc == base_netloc and full_url not in seen_urls:
                title = link.get("title", "").strip() or link.get_text(strip=True)
                if title and len(title) > 5:
                    if not re.search(r"\.(pdf|jpg|png|gif|jpeg|svg|css|js)$", full_url, re.IGNORECASE):
                        articles.append({"url": full_url, "title": title})
                        seen_urls.add(full_url)

            if len(articles) >= max_links:
                break

        print(f"Found {len(articles)} article links")
        return articles

    def _scrape_article(self, article_info):
        url = article_info["url"]
        title = article_info["title"]
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            r = requests.get(url, headers=headers, timeout=30)
            soup = BeautifulSoup(r.text, "html.parser")

            title_tag = soup.find("h1")
            page_title = title_tag.get_text(strip=True) if title_tag else title

            paragraphs = []
            content_div = (
                soup.find("div", class_=re.compile(r"article|content|body|text", re.I))
                or soup.find("div", id=re.compile(r"article|content|body|text", re.I))
            )

            if content_div:
                for p in content_div.find_all("p"):
                    text = p.get_text(strip=True)
                    if text and len(text) > 10:
                        paragraphs.append(text)

            if not paragraphs:
                for p in soup.find_all("p"):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20: 
                        paragraphs.append(text)

            full_text = "\n\n".join(paragraphs)

            if not full_text:
                print(f"No text found for: {url}")
                return None

            print(f"Scraped: {url}")
            return {
                "url": url,
                "title": page_title,
                "text": full_text,
                "num_paragraphs": len(paragraphs),
            }

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def scrape_website(self, max_articles=500):
        print("Starting scraper")
        all_article_links = []
        for url in self.base_urls:
            links = self._get_article_links(url, max_links=max_articles)
            if links:
                all_article_links.extend(links)

        if not all_article_links:
            print("No article links found across any sites. Exiting.")
            return []

        scraped_articles = []
        
        print(f"Firing multithreaded requests for {len(all_article_links)} links...")
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(self._scrape_article, all_article_links)
            
        for article in results:
            if article:
                scraped_articles.append(article)

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(scraped_articles, f, ensure_ascii=False, indent=2)
            
        ellapsed = time.time() - start_time
        print(f"Done Scraped {len(scraped_articles)} articles in {ellapsed:.2f} seconds")

        return scraped_articles
    
    def process_and_store_data(self, db_path="Data Processing Pipeline/chroma_db"):
        print(f"Loading data from {self.output_file}")
        if not os.path.exists(self.output_file):
            print(f"Error: {self.output_file} does not exist.")
            return

        with open(self.output_file, "r", encoding="utf-8") as f:
            articles = json.load(f)

        if not articles:
            print("No data found in json file.")
            return

        print("Initializing embedding model")
        model = SentenceTransformer('intfloat/multilingual-e5-small')
        
        print("Initializing vector database")
        client = chromadb.PersistentClient(path=db_path)
        
        collection = client.get_or_create_collection(
            name="website_articles",
            metadata={"hnsw:space": "cosine"}
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        documents = []
        metadatas = []
        ids = []
        
        print("Processing articles")
        for i, article in enumerate(articles):
            if not article.get("text"):
                continue
                
            chunks = text_splitter.split_text(article['text'])
            
            for j, chunk in enumerate(chunks):
                content = f"Title: {article['title']}\n\nContent: {chunk}"
                documents.append(content)
                metadatas.append({
                    "url": article["url"],
                    "title": article["title"],
                    "chunk_id": j
                })
                ids.append(f"doc_{i}_chunk_{j}")

        if not documents:
            print("No valid documents to embed.")
            return
            
        print(f"Generating embeddings for {len(documents)} documents")
        embeddings = model.encode(documents, normalize_embeddings=True, batch_size=32)

        print("Storing in ChromaDB")
        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )

        print(f"Successfully stored {len(documents)} chunks in vector DB.")

    def scrape_and_store(self):
        self.scrape_website()
        self.process_and_store_data()
