"""
Собственный парсер сайтов на основе Readability алгоритма
Автор: Дмитрий Исаев
"""
import cloudscraper
from bs4 import BeautifulSoup
import re


class CustomParser:
    def __init__(self):
        self.session = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'ru-RU,ru;q=0.9',
        })

    def parse_url(self, url):
        """Главная функция: парсит URL и возвращает чистый текст"""
        try:
            print(f"[PARSER] 📖 Читаю: {url}")

            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')

            # Упрощённое удаление мусора
            self._remove_junk(soup)

            # Пробуем несколько стратегий поиска контента
            main_content = None

            # Стратегия 1: Ищем по семантическим тегам
            for tag in ['article', 'main', '[role="main"]']:
                element = soup.select_one(tag)
                if element and len(element.get_text(strip=True)) > 200:
                    main_content = element
                    print(f"[PARSER] 🎯 Нашёл через тег: {tag}")
                    break

            # Стратегия 2: Ищем по ID/классам
            if not main_content:
                good_ids = ['content', 'bodyContent', 'article', 'main-content',
                            'mw-content-text', 'content-body']
                good_classes = ['article-body', 'post-content', 'entry-content',
                                'mw-body', 'b-article__body']

                for selector in good_ids + good_classes:
                    element = soup.find(id=selector) or soup.find(class_=selector)
                    if element and len(element.get_text(strip=True)) > 200:
                        main_content = element
                        print(f"[PARSER] 🎯 Нашёл через селектор: {selector}")
                        break

            # Стратегия 3: Readability-алгоритм (fallback)
            if not main_content:
                main_content = self._find_by_score(soup)
                if main_content:
                    print(f"[PARSER] 🎯 Нашёл через scoring")

            # Стратегия 4: Берём body (последний fallback)
            if not main_content:
                main_content = soup.find('body')
                print(f"[PARSER] ⚠️ Fallback на body")

            if main_content:
                text = self._extract_text(main_content)
                text = self._clean_text(text)
                print(f"[PARSER] ✅ Извлечено {len(text)} символов")
                return text

            return None

        except Exception as e:
            print(f"[PARSER Error] {url}: {e}")
            return None

    def _remove_junk(self, soup):
        """Удаляем ТОЛЬКО явный мусор"""
        # Теги — точно не контент
        for tag in soup.find_all(['script', 'style', 'nav', 'iframe', 'noscript',
                                  'form', 'svg', 'canvas']):
            tag.decompose()

        # Элементы с явными "плохими" классами
        bad_classes = ['cookie', 'popup', 'modal', 'sidebar', 'advertisement',
                       'social-share', 'comments']

        for element in soup.find_all(True):
            classes = ' '.join(element.get('class', []) or [])
            if any(bad in classes.lower() for bad in bad_classes):
                element.decompose()

    def _find_by_score(self, soup):
        """Readability-алгоритм (упрощённый)"""
        candidates = []

        for element in soup.find_all(['div', 'section', 'article']):
            text = element.get_text(strip=True)
            if len(text) < 200:
                continue

            score = 0

            # Длина текста
            score += min(len(text) / 100, 15)

            # Количество параграфов
            paragraphs = element.find_all('p')
            score += len(paragraphs) * 2

            # Плотность ссылок (штраф)
            links = element.find_all('a')
            link_text = sum(len(a.get_text(strip=True)) for a in links)
            if len(text) > 0 and link_text / len(text) > 0.3:
                score -= 10

            # Положительные классы
            classes = ' '.join(element.get('class', []) or []).lower()
            if any(x in classes for x in ['article', 'content', 'post', 'text']):
                score += 10

            candidates.append((score, element))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    def _extract_text(self, element):
        """Извлекаем текст"""
        paragraphs = element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])

        if paragraphs:
            texts = [p.get_text(strip=True) for p in paragraphs
                     if len(p.get_text(strip=True)) > 15]
            return '\n\n'.join(texts)

        return element.get_text(separator='\n', strip=True)

    def _clean_text(self, text):
        """Очистка"""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 15]
        return '\n'.join(lines).strip()