import ast
import html
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


LEAFLET_URL = (
    "https://eapp.emart.com/leaflet/leafletView_EL.do"
    "?trcknCode=main_leaflet"
    "&referer=https%3A%2F%2Feapp.emart.com%2Fwebapp%2Fproduct%2Fflyer%3Fnull"
)

KST = timezone(timedelta(hours=9))


class LeafletParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.image_urls = []
        self._in_detail = False
        self._detail_depth = 0
        self._detail_chunks = []
        self.pages = []

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag == "img":
            src = attr.get("data-src") or attr.get("src") or ""
            if "/upload/news_leaflet/" in src and src not in self.image_urls:
                self.image_urls.append(src)

        classes = attr.get("class", "")
        if tag == "div" and "img_detail_txt" in classes:
            self._in_detail = True
            self._detail_depth = 1
            self._detail_chunks = []
        elif self._in_detail:
            self._detail_depth += 1

    def handle_endtag(self, tag):
        if not self._in_detail:
            return
        self._detail_depth -= 1
        if self._detail_depth == 0:
            text = re.sub(r"\s+", " ", " ".join(self._detail_chunks)).strip()
            self.pages.append(html.unescape(text))
            self._in_detail = False

    def handle_data(self, data):
        if self._in_detail:
            self._detail_chunks.append(data)


def load_standard_ingredients(base_dir):
    tree = ast.parse((base_dir / "extract_ingredients.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(getattr(target, "id", None) == "standard_ingredients" for target in node.targets):
                return ast.literal_eval(node.value)
    raise RuntimeError("standard_ingredients not found")


def fetch_leaflet(url=LEAFLET_URL):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=20) as res:
        return res.read().decode("utf-8", errors="replace")


def extract_food_items_with_llm(text):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from openai import OpenAI
    except ImportError:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []

    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract grocery food items from Korean supermarket leaflet text. "
                    "Return only a JSON array of strings. Keep the leaflet's product wording and modifiers. "
                    "Exclude prices, dates, card benefits, coupons, event copy, and non-food items."
                ),
            },
            {
                "role": "user",
                "content": (
                    "전단 1면 텍스트에서 식재료/식품 상품명만 뽑아주세요. "
                    "예: 체리, 신비복숭아, 한우 헤비 스테이크 3종처럼 수식어와 상품명은 유지합니다.\n\n"
                    f"{text}"
                ),
            },
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL)
    try:
        items = json.loads(content)
    except json.JSONDecodeError:
        return []
    return [item.strip() for item in items if isinstance(item, str) and item.strip()]


def ingredient_aliases():
    # ponytail: flyer text heuristic; add aliases here if Emart phrasing causes misses.
    return {
        "돼지고기": ["한돈", "돈목심", "앞다리", "뒷다리", "돼지", "폭립"],
        "소고기": ["한우", "와규", "우삼겹", "차돌", "등심", "안심", "채끝", "불고기", "국거리"],
        "닭고기": ["통닭", "백숙", "삼계탕", "치킨", "닭"],
        "스팸(햄)": ["스팸", "햄"],
        "새우": ["감바스"],
        "참치": ["참다랑어", "살코기참치"],
        "김치": ["포기김치", "종가 김치"],
        "버섯": ["새송이버섯", "머쉬룸"],
        "콩나물/숙주": ["숙주", "콩나물"],
        "밥(쌀)": ["백미밥", "잡곡밥", "햇반", "즉석밥", "쌀"],
        "카레가루(카레)": ["카레"],
        "짜장가루(춘장)": ["짜장"],
        "라면": ["봉지면"],
        "불닭소스": ["불닭소스"],
        "요거트": ["그릭요거트"],
        "우유": ["우유"],
        "치즈": ["치즈"],
        "두부": ["두부"],
        "토마토": ["토마토", "토마주르"],
        "단호박": ["단호박"],
        "감자": ["감자"],
        "블루베리": ["블루베리"],
        "딸기": ["딸기"],
        "사과": ["사과"],
        "레몬(레몬즙)": ["유자", "레몬"],
}


def term_matches(text, term):
    if len(term) <= 1 or term in {"가지", "무"}:
        return re.search(rf"(?<![가-힣]){re.escape(term)}(?![가-힣])", text) is not None
    return term in text


def match_ingredients(text, standard_ingredients):
    standard = set(standard_ingredients)
    aliases = ingredient_aliases()
    matched = {}

    for name in standard_ingredients:
        terms = [name] + aliases.get(name, [])
        pages = [
            page_no
            for page_no, page_text in enumerate(text, start=1)
            if any(term and term_matches(page_text, term) for term in terms)
        ]
        if pages:
            matched[name] = pages

    return [
        {"name": name, "pages": matched[name]}
        for name in standard_ingredients
        if name in matched
    ]


def scrape(url=LEAFLET_URL):
    base_dir = Path(__file__).resolve().parent
    parser = LeafletParser()
    parser.feed(fetch_leaflet(url))
    first_page_texts = parser.pages[:1]
    first_page_images = parser.image_urls[:1]

    pages = [
        {
            "page": i + 1,
            "image_url": first_page_images[i] if i < len(first_page_images) else "",
            "text": text,
        }
        for i, text in enumerate(first_page_texts)
    ]

    data = {
        "source_url": url,
        "fetched_at": datetime.now(KST).isoformat(timespec="seconds"),
        "image_urls": first_page_images,
        "pages": pages,
        "llm_food_items": extract_food_items_with_llm(first_page_texts[0]) if first_page_texts else [],
        "matched_ingredients": match_ingredients(first_page_texts, load_standard_ingredients(base_dir)),
    }
    return data


def self_check():
    data = scrape()
    assert len(data["image_urls"]) == 1, f"expected 1 leaflet image, got {len(data['image_urls'])}"
    assert len(data["pages"]) == 1, f"expected 1 leaflet text page, got {len(data['pages'])}"
    assert data["matched_ingredients"], "expected at least one matched ingredient"
    assert isinstance(data["llm_food_items"], list), "expected llm_food_items list"
    print("self-check passed")


def main():
    if "--self-check" in sys.argv:
        self_check()
        return

    base_dir = Path(__file__).resolve().parent
    data = scrape()
    (base_dir / "emart_leaflet.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    print(
        "Generated emart_leaflet.json with "
        f"{len(data['pages'])} pages and {len(data['matched_ingredients'])} ingredients."
    )


if __name__ == "__main__":
    main()
