import warnings
from typing import Iterator, Literal, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_env


class FireCrawlLoader(BaseLoader):
    """
    FireCrawlLoader document loader integration

    Setup:
        Install ``firecrawl-py``,``langchain_community`` and set environment variable ``FIRECRAWL_API_KEY``.

        .. code-block:: bash

            pip install -U firecrawl-py langchain_community
            export FIRECRAWL_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import FireCrawlLoader

            loader = FireCrawlLoader(
                url = "https://firecrawl.dev",
                mode = "crawl"
                # other params = ...
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Introducing [Smart Crawl!](https://www.firecrawl.dev/smart-crawl)
             Join the waitlist to turn any web
            {'ogUrl': 'https://www.firecrawl.dev/', 'title': 'Home - Firecrawl', 'robots': 'follow, index', 'ogImage': 'https://www.firecrawl.dev/og.png?123', 'ogTitle': 'Firecrawl', 'sitemap': {'lastmod': '2024-08-12T00:28:16.681Z', 'changefreq': 'weekly'}, 'keywords': 'Firecrawl,Markdown,Data,Mendable,Langchain', 'sourceURL': 'https://www.firecrawl.dev/', 'ogSiteName': 'Firecrawl', 'description': 'Firecrawl crawls and converts any website into clean markdown.', 'ogDescription': 'Turn any website into LLM-ready data.', 'pageStatusCode': 200, 'ogLocaleAlternate': []}

    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Introducing [Smart Crawl!](https://www.firecrawl.dev/smart-crawl)
             Join the waitlist to turn any web
            {'ogUrl': 'https://www.firecrawl.dev/', 'title': 'Home - Firecrawl', 'robots': 'follow, index', 'ogImage': 'https://www.firecrawl.dev/og.png?123', 'ogTitle': 'Firecrawl', 'sitemap': {'lastmod': '2024-08-12T00:28:16.681Z', 'changefreq': 'weekly'}, 'keywords': 'Firecrawl,Markdown,Data,Mendable,Langchain', 'sourceURL': 'https://www.firecrawl.dev/', 'ogSiteName': 'Firecrawl', 'description': 'Firecrawl crawls and converts any website into clean markdown.', 'ogDescription': 'Turn any website into LLM-ready data.', 'pageStatusCode': 200, 'ogLocaleAlternate': []}

    """  # noqa: E501

    def __init__(
        self,
        url: str,
        *,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        mode: Literal["crawl", "scrape", "map", "extract", "search"] = "crawl",
        params: Optional[dict] = None,
    ):
        """Initialize with API key and url.

        Args:
            url: The url to be crawled.
            api_key: The Firecrawl API key. If not specified will be read from env var
                FIRECRAWL_API_KEY. Get an API key
            api_url: The Firecrawl API URL. If not specified will be read from env var
                FIRECRAWL_API_URL or defaults to https://api.firecrawl.dev.
            mode: The mode to run the loader in. Default is "crawl".
                 Options include "scrape" (single url),
                 "crawl" (all accessible sub pages),
                 "map" (returns list of links that are semantically related).
                 "extract" (extracts structured data from a page).
                 "search" (search for data across the web).
            params: The parameters to pass to the Firecrawl API.
                Examples include crawlerOptions.
                For more details, visit: https://github.com/mendableai/firecrawl-py
        """

        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )
        if mode not in ("crawl", "scrape", "search", "map", "extract", "search"):
            raise ValueError(
                f"""Invalid mode '{mode}'.
                Allowed: 'crawl', 'scrape', 'search', 'map', 'extract', 'search'."""
            )

        if not url:
            raise ValueError("Url must be provided")

        api_key = api_key or get_from_env("api_key", "FIRECRAWL_API_KEY")
        self.firecrawl = FirecrawlApp(api_key=api_key, api_url=api_url)
        self.url = url
        self.mode = mode
        self.params = params or {}

    def lazy_load(self) -> Iterator[Document]:
        if self.mode == "scrape":
            # Create ScrapeOptions if scrapeOptions provided
            scrape_options = None
            if "scrapeOptions" in self.params:
                try:
                    from firecrawl import ScrapeOptions
                    scrape_options = ScrapeOptions(**self.params["scrapeOptions"])
                except ImportError:
                    pass
            
            firecrawl_docs = [
                self.firecrawl.scrape_url(
                    self.url, 
                    scrape_options=scrape_options
                )
            ]
        elif self.mode == "crawl":
            if not self.url:
                raise ValueError("URL is required for crawl mode")
            
            # Extract parameters for crawl_url method
            crawl_kwargs = {}
            
            # Map parameters to correct names
            if 'maxDepth' in self.params:
                crawl_kwargs['max_depth'] = self.params['maxDepth']
            if 'limit' in self.params:
                crawl_kwargs['limit'] = self.params['limit']
            if 'includePaths' in self.params:
                crawl_kwargs['include_paths'] = self.params['includePaths']
            if 'excludePaths' in self.params:
                crawl_kwargs['exclude_paths'] = self.params['excludePaths']
            if 'allowExternalLinks' in self.params:
                crawl_kwargs['allow_external_links'] = self.params['allowExternalLinks']
            if 'allowBackwardLinks' in self.params:
                crawl_kwargs['allow_backward_links'] = self.params['allowBackwardLinks']
            if 'ignoreSitemap' in self.params:
                crawl_kwargs['ignore_sitemap'] = self.params['ignoreSitemap']
            
            # Handle scrapeOptions
            if 'scrapeOptions' in self.params:
                try:
                    from firecrawl import ScrapeOptions
                    crawl_kwargs['scrape_options'] = ScrapeOptions(**self.params['scrapeOptions'])
                except ImportError:
                    # Fallback: pass as dict
                    crawl_kwargs['scrape_options'] = self.params['scrapeOptions']
            
            crawl_response = self.firecrawl.crawl_url(
                self.url, **crawl_kwargs
            )
            firecrawl_docs = crawl_response.data if hasattr(crawl_response, 'data') else []
        elif self.mode == "map":
            if not self.url:
                raise ValueError("URL is required for map mode")
            firecrawl_docs = self.firecrawl.map_url(self.url, params=self.params)
        elif self.mode == "extract":
            if not self.url:
                raise ValueError("URL is required for extract mode")
            firecrawl_docs = [
                str(self.firecrawl.extract([self.url], params=self.params))
            ]
        elif self.mode == "search":
            firecrawl_docs = self.firecrawl.search(
                query=self.params.get("query"),
                params=self.params,
            )
        else:
            raise ValueError(
                f"""Invalid mode '{self.mode}'.
                Allowed: 'crawl', 'scrape', 'map', 'extract', 'search'."""
            )
        for doc in firecrawl_docs:
            if self.mode == "map" or self.mode == "extract":
                page_content = doc
                metadata = {}
            else:
                page_content = (
                    getattr(doc, "markdown", None) or getattr(doc, "html", None) or getattr(doc, "rawHtml", "")
                )
                metadata = getattr(doc, "metadata", {}) or {}
            if not page_content:
                continue
            yield Document(
                page_content=page_content,
                metadata=metadata,
            )
