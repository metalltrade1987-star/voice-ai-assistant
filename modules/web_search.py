"""Поиск в интернете: DuckDuckGo + Погода + Новости + Crawl4AI для чтения сайтов"""
import requests
import asyncio
from datetime import datetime
from duckduckgo_search import DDGS


class WebSearcher:
    def __init__(self):
        self.ddgs = DDGS()
        print("[WEB] Модуль поиска инициализирован (с Crawl4AI)")

    def search(self, query, max_results=5):
        """Общий поиск через DuckDuckGo"""
        try:
            print(f"[WEB] Поиск: '{query}'")
            results = self.ddgs.text(query, region="ru-ru", max_results=max_results)

            if not results:
                return "Ничего не найдено.", []

            answer = f"По запросу '{query}' нашёл:\n\n"
            urls = []
            for i, r in enumerate(results, 1):
                title = r.get('title', 'Без названия')
                body = r.get('body', '')
                href = r.get('href', '')
                urls.append(href)
                answer += f"{i}. {title}\n   {body}\n   {href}\n\n"

            return answer.strip(), urls
        except Exception as e:
            print(f"[WEB Error] search: {e}")
            return f"Ошибка поиска: {e}", []

    def _read_url(self, url):
        """
        🔥 Читает сайт через Crawl4AI (локально, бесплатно, без лимитов)
        """
        try:
            print(f"[WEB] 📖 Читаю через Crawl4AI: {url}")

            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

            async def crawl():
                browser_config = BrowserConfig(
                    headless=True,
                    verbose=False
                )

                # 🔥 Только рабочие параметры для Crawl4AI 0.9.0
                crawl_config = CrawlerRunConfig(
                    cache_mode=CacheMode.DISABLED,
                    word_count_threshold=50,
                    css_selector="main, article, [role='main'], #content, #bodyContent, .content, .post-content, .article-content, .entry-content",
                    exclude_external_links=True,
                    exclude_social_media_links=True,
                    exclude_internal_links=True,
                )

                async with AsyncWebCrawler(config=browser_config) as crawler:
                    result = await crawler.arun(url=url, config=crawl_config)

                    if result.success:
                        return result.markdown
                    else:
                        print(f"[WEB Error] Crawl4AI не смог прочитать: {result.error_message}")
                        return None

            text = asyncio.run(crawl())

            if text:
                # Простая очистка
                text = self._clean_markdown(text)

                print(f"[WEB] ✅ Crawl4AI прочитал {len(text)} символов")
                return text

            return None

        except Exception as e:
            print(f"[WEB Error] Crawl4AI ошибка для {url}: {e}")
            return None

    def _clean_markdown(self, text):
        """Простая очистка Markdown"""
        import re

        # Удаляем Markdown ссылки
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        # Удаляем Markdown изображения
        text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)

        # Удаляем заголовки
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

        # Удаляем HTML теги
        text = re.sub(r'<[^>]+>', '', text)

        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _smart_query(self, query):
        """Уточняет поисковый запрос для лучших результатов"""
        query_lower = query.lower()

        # Для фильмов — добавляем Кинопоиск
        if any(word in query_lower for word in ['фильм', 'сериал', 'кино', 'рейтинг фильма']):
            if 'кинопоиск' not in query_lower and 'imdb' not in query_lower:
                query += " Кинопоиск"

        # Для книг — добавляем Лабиринт
        if any(word in query_lower for word in ['книга', 'автор', 'произведение']):
            if 'лабиринт' not in query_lower:
                query += " Лабиринт"

        return query

    def deep_search(self, query, max_sites=3):
        # 🔥 Уточняем запрос
        query = self._smart_query(query)
        """
        🔥 ГЛУБОКИЙ ПОИСК: находит сайты и читает их через Crawl4AI
        """
        try:
            print(f"[WEB] 🔥 Глубокий поиск: '{query}'")

            # 1. Ищем URL через DuckDuckGo
            results = self.ddgs.text(query, region="ru-ru", max_results=max_sites * 2)

            if not results:
                return "Ничего не найдено.", []

            # 2. Читаем контент каждого сайта
            sources = []
            contents = []

            for r in results:
                if len(sources) >= max_sites:
                    break

                url = r.get('href', '')
                title = r.get('title', '')
                snippet = r.get('body', '')

                if not url:
                    continue

                # Читаем через Crawl4AI
                content = self._read_url(url)

                if content and len(content) > 200:
                    contents.append(f"=== {title} ===\n{content}\n")
                    sources.append({
                        'title': title,
                        'url': url,
                        'source_type': 'crawl4ai',
                        'content_length': len(content)
                    })
                    print(f"[WEB] ✅ Успешно прочитано: {title}")
                else:
                    # Fallback: используем сниппет
                    print(f"[WEB] ⚠️ Crawl4AI не смог прочитать, использую сниппет: {title}")
                    contents.append(f"=== {title} ===\n{snippet}\n")
                    sources.append({
                        'title': title,
                        'url': url,
                        'source_type': 'snippet',
                        'content_length': len(snippet)
                    })

            if not contents:
                return "Не удалось получить информацию ни из одного источника.", []

            combined = "\n\n".join(contents)
            return combined, sources

        except Exception as e:
            print(f"[WEB Error] deep_search: {e}")
            return f"Ошибка глубокого поиска: {e}", []

    def get_news(self, query, max_results=5):
        """Поиск новостей через DuckDuckGo"""
        try:
            print(f"[WEB] Новости: '{query}'")
            results = self.ddgs.news(query, region="ru-ru", max_results=max_results)

            if not results:
                results = self.ddgs.news(query, region="wt-wt", max_results=max_results)

            if not results:
                return "Новостей не найдено.", []

            answer = f"Свежие новости по запросу '{query}':\n\n"
            urls = []
            for i, r in enumerate(results, 1):
                title = r.get('title', 'Без названия')
                body = r.get('body', '')
                date = r.get('date', '')
                source = r.get('source', '')
                href = r.get('url', '')
                urls.append(href)
                answer += f"{i}. [{source}] {title}\n   {body}\n   {date}\n\n"

            return answer.strip(), urls
        except Exception as e:
            print(f"[WEB Error] news: {e}")
            return f"Ошибка поиска новостей: {e}", []

    def get_weather(self, city="Москва"):
        """Погода через Open-Meteo"""
        try:
            print(f"[WEB] Погода в городе: {city}")

            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=ru"
            geo_resp = requests.get(geo_url, timeout=15).json()

            if "results" not in geo_resp or not geo_resp["results"]:
                return f"Не удалось найти город {city}.", []

            lat = geo_resp["results"][0]["latitude"]
            lon = geo_resp["results"][0]["longitude"]
            country = geo_resp["results"][0].get("country", "")

            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
                f"&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum"
                f"&timezone=auto&forecast_days=3"
            )
            weather_resp = requests.get(weather_url, timeout=15).json()

            current = weather_resp["current"]
            daily = weather_resp["daily"]

            weather_codes = {
                0: "ясно", 1: "преимущественно ясно", 2: "переменная облачность",
                3: "пасмурно", 45: "туман", 48: "изморозь",
                51: "небольшая морось", 53: "морось", 55: "сильная морось",
                61: "небольшой дождь", 63: "дождь", 65: "сильный дождь",
                71: "небольшой снег", 73: "снег", 75: "сильный снег",
                80: "небольшой ливень", 81: "ливень", 82: "сильный ливень",
                95: "гроза", 96: "гроза с градом", 99: "сильная гроза с градом"
            }

            temp = current["temperature_2m"]
            humidity = current["relative_humidity_2m"]
            wind = current["wind_speed_10m"]
            code = current["weather_code"]
            weather_desc = weather_codes.get(code, f"код {code}")

            answer = f"Погода в городе {city}"
            if country:
                answer += f", {country}"
            answer += f":\n\n"
            answer += f"🌡️ Температура: {temp}°C\n"
            answer += f"💧 Влажность: {humidity}%\n"
            answer += f"💨 Ветер: {wind} м/с\n"
            answer += f"☁️ Состояние: {weather_desc}\n\n"

            answer += "Прогноз на 3 дня:\n"
            for i in range(min(3, len(daily["time"]))):
                date = daily["time"][i]
                t_max = daily["temperature_2m_max"][i]
                t_min = daily["temperature_2m_min"][i]
                precip = daily["precipitation_sum"][i]
                day_code = daily["weather_code"][i]
                day_desc = weather_codes.get(day_code, f"код {day_code}")

                try:
                    dt = datetime.strptime(date, "%Y-%m-%d")
                    date_str = dt.strftime("%d.%m")
                except:
                    date_str = date

                answer += f"  {date_str}: {t_min}°...{t_max}°C, {day_desc}"
                if precip > 0:
                    answer += f", осадки {precip} мм"
                answer += "\n"

            return answer.strip(), []
        except Exception as e:
            print(f"[WEB Error] weather: {e}")
            return f"Ошибка получения погоды: {e}", []