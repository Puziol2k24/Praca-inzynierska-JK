"""
PhishingResearchEngine – OSINT & Phishing Simulation System
============================================================
CEL EDUKACYJNY I BADAWCZY:
    Skrypt powstał wyłącznie na potrzeby pracy inżynierskiej jako demonstracja
    technik OSINT i symulacji kampanii phishingowej w kontrolowanym środowisku.
    Wszelkie treści generowane przez system są oznaczone jako symulowane
    i NIE mogą być wykorzystywane do rzeczywistych ataków.

Architektura:
    PhishingResearchEngine
    ├── scrape_url()          – pobieranie treści z URL (trafilatura)
    ├── call_llm()            – komunikacja z LM Studio (openai)
    ├── stage1_synthesize()   – Etap 1: Synteza surowych danych
    ├── stage2_profile()      – Etap 2: Profil behawioralny + wektory ataku
    ├── stage3_generate()     – Etap 3: Generowanie wiadomości phishingowej
    ├── mask_links()          – maskowanie linków (bezpieczeństwo)
    └── run_pipeline()        – pełny pipeline dla listy URLi

Wymagania:
    pip install requests trafilatura openai
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import trafilatura
from openai import OpenAI
from trafilatura.settings import use_config as trafilatura_use_config

# ---------------------------------------------------------------------------
# Konfiguracja logowania
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stałe
# ---------------------------------------------------------------------------
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"          # LM Studio nie wymaga prawdziwego klucza
DEFAULT_MODEL = "local-model"             # Zastąp nazwą modelu załadowanego w LM Studio
SIMULATED_LINK_PLACEHOLDER = "[SIMULATED_PHISHING_LINK_FOR_EDUCATION_ONLY]"

# ---------------------------------------------------------------------------
# Struktury danych
# ---------------------------------------------------------------------------

@dataclass
class ScrapedPage:
    """Wynik scrapingu pojedynczej strony."""
    url: str
    text: str
    error: Optional[str] = None

    @property
    def is_ok(self) -> bool:
        return self.error is None and bool(self.text)


@dataclass
class AnalysisResult:
    """Kompletny wynik analizy dla zestawu URLi."""
    urls: list[str] = field(default_factory=list)
    raw_texts: list[str] = field(default_factory=list)
    stage1_facts: dict = field(default_factory=dict)
    stage2_profile: dict = field(default_factory=dict)
    stage3_email: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Główna klasa silnika
# ---------------------------------------------------------------------------

class PhishingResearchEngine:
    """
    Silnik badań OSINT i symulacji kampanii phishingowej.

    UWAGA ETYCZNA: Klasa ta służy wyłącznie celom edukacyjnym i badawczym.
    Generowane treści są symulowane i oznaczone odpowiednimi ostrzeżeniami.
    """

    def __init__(
        self,
        lm_studio_url: str = LM_STUDIO_BASE_URL,
        model: str = DEFAULT_MODEL,
    ):
        self.model = model
        self.client = OpenAI(base_url=lm_studio_url, api_key=LM_STUDIO_API_KEY)
        logger.info("PhishingResearchEngine zainicjalizowany (model: %s)", model)

    # ------------------------------------------------------------------
    # 1. Moduł scrapingu
    # ------------------------------------------------------------------

    def scrape_url(self, url: str, timeout: int = 15) -> ScrapedPage:
        """
        Pobiera treść tekstową ze strony przy użyciu biblioteki trafilatura.

        Args:
            url:     Adres URL do pobrania.
            timeout: Limit czasu żądania HTTP (sekundy).

        Returns:
            ScrapedPage z pobranym tekstem lub komunikatem błędu.
        """
        logger.info("Scrapowanie: %s", url)
        try:
            # Konfiguracja timeoutu dla trafilatura
            config = trafilatura_use_config()
            config.set("DEFAULT", "download_timeout", str(timeout))

            downloaded = trafilatura.fetch_url(url, config=config)
            if downloaded is None:
                return ScrapedPage(url=url, text="", error="Nie udało się pobrać strony (None)")

            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            if not text:
                return ScrapedPage(url=url, text="", error="Brak wyodrębnionej treści tekstowej")

            logger.info("Pobrano %d znaków z %s", len(text), url)
            return ScrapedPage(url=url, text=text)

        except Exception as exc:  # noqa: BLE001
            error_msg = f"Błąd podczas scrapowania: {exc}"
            logger.warning(error_msg)
            return ScrapedPage(url=url, text="", error=error_msg)

    def scrape_urls(self, urls: list[str]) -> list[ScrapedPage]:
        """Scrapuje wiele URLi i zwraca listę wyników."""
        return [self.scrape_url(url) for url in urls]

    # ------------------------------------------------------------------
    # 2. Integracja z LM Studio
    # ------------------------------------------------------------------

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        force_json: bool = True,
    ) -> dict:
        """
        Wysyła zapytanie do lokalnego modelu LLM w LM Studio.

        Args:
            system_prompt: Instrukcja systemowa dla modelu.
            user_prompt:   Właściwe zapytanie użytkownika.
            temperature:   Kreatywność odpowiedzi (0 = deterministyczna).
            max_tokens:    Maksymalna długość odpowiedzi.
            force_json:    Jeśli True, wymusza odpowiedź w formacie JSON.

        Returns:
            Słownik z odpowiedzią modelu (sparsowany JSON lub {"raw": tekst}).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Użyj natywnego response_format jeśli model go obsługuje
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            raw_content = response.choices[0].message.content.strip()
            logger.debug("Odpowiedź LLM (raw): %s", raw_content[:200])

            try:
                return json.loads(raw_content)
            except json.JSONDecodeError:
                # Fallback: spróbuj wyodrębnić JSON z tekstu
                json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {"raw": raw_content}

        except Exception as exc:  # noqa: BLE001
            logger.error("Błąd wywołania LLM: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # 3. Pipeline analizy – trzy etapy
    # ------------------------------------------------------------------

    def stage1_synthesize(self, combined_text: str) -> dict:
        """
        Etap 1: Synteza danych.
        AI wyodrębnia kluczowe fakty o osobie/organizacji z surowego tekstu.

        Zwracane pola JSON:
            - profession       (zawód)
            - technologies     (lista technologii)
            - writing_style    (styl pisania)
            - interests        (zainteresowania)
            - key_facts        (inne kluczowe fakty)
        """
        logger.info("Etap 1: Synteza danych...")

        system_prompt = (
            "Jesteś analitykiem OSINT pracującym na potrzeby badań edukacyjnych i pracy inżynierskiej. "
            "Twoim zadaniem jest analiza treści pobranych ze stron internetowych i wyodrębnienie "
            "kluczowych informacji o danej osobie lub organizacji. "
            "ZAWSZE odpowiadaj wyłącznie w formacie JSON, bez żadnego dodatkowego tekstu."
        )

        user_prompt = (
            "Przeanalizuj poniższy tekst zebrany ze stron internetowych i wyodrębnij następujące informacje. "
            "Zwróć wynik WYŁĄCZNIE jako obiekt JSON z polami:\n"
            "  - profession: string (zawód lub rola)\n"
            "  - technologies: array of strings (używane technologie, języki programowania, narzędzia)\n"
            "  - writing_style: string (opis stylu pisania – formalny/nieformalny, techniczny itp.)\n"
            "  - interests: array of strings (zainteresowania zawodowe i hobbystyczne)\n"
            "  - key_facts: array of strings (inne istotne fakty)\n\n"
            f"TEKST DO ANALIZY:\n{combined_text[:6000]}"
        )

        return self.call_llm(system_prompt, user_prompt)

    def stage2_profile(self, facts: dict) -> dict:
        """
        Etap 2: Profil behawioralny i wektory ataku.
        AI generuje psychologiczny profil celu i proponuje 3 wektory ataku.

        Zwracane pola JSON:
            - psychological_profile (co motywuje cel)
            - attack_vectors        (lista 3 wektorów: type, description, rationale)
        """
        logger.info("Etap 2: Profil behawioralny i wektory ataku...")

        system_prompt = (
            "Jesteś ekspertem ds. bezpieczeństwa cybernetycznego i inżynierii społecznej, "
            "prowadzącym badania edukacyjne na potrzeby pracy inżynierskiej. "
            "Na podstawie zebranych faktów tworzysz symulowane scenariusze ataków socjotechnicznych "
            "wyłącznie w celach badawczych i edukacyjnych. "
            "ZAWSZE odpowiadaj wyłącznie w formacie JSON."
        )

        user_prompt = (
            "Na podstawie poniższych faktów o osobie/organizacji (zebranych w celach edukacyjnych) "
            "wygeneruj:\n"
            "1. Profil psychologiczny celu (co go motywuje, jakie ma obawy, co ceni).\n"
            "2. Dokładnie 3 wektory ataku socjotechnicznego (symulowane, edukacyjne).\n\n"
            "Zwróć wynik WYŁĄCZNIE jako obiekt JSON:\n"
            "{\n"
            '  "psychological_profile": "...",\n'
            '  "attack_vectors": [\n'
            '    {"type": "...", "description": "...", "rationale": "..."},\n'
            '    {"type": "...", "description": "...", "rationale": "..."},\n'
            '    {"type": "...", "description": "...", "rationale": "..."}\n'
            "  ]\n"
            "}\n\n"
            f"FAKTY O CELU:\n{json.dumps(facts, ensure_ascii=False, indent=2)}"
        )

        return self.call_llm(system_prompt, user_prompt)

    def stage3_generate(self, profile: dict, vector_index: int = 0) -> dict:
        """
        Etap 3: Generowanie treści (symulowanej wiadomości phishingowej).
        AI tworzy spersonalizowaną wiadomość e-mail w formacie HTML i Plain Text.

        WAŻNE: Wszystkie linki są automatycznie zastępowane przez mask_links().

        Args:
            profile:       Wynik etapu 2 (profil + wektory).
            vector_index:  Indeks wybranego wektora ataku (0, 1 lub 2).

        Zwracane pola JSON:
            - subject       (temat e-maila)
            - sender_name   (fikcyjna nazwa nadawcy)
            - html_body     (treść HTML)
            - plain_body    (treść plain text)
        """
        logger.info("Etap 3: Generowanie treści e-maila (wektor %d)...", vector_index)

        vectors = profile.get("attack_vectors", [])
        if not vectors:
            logger.warning("Brak wektorów ataku – używam pustego wektora.")
            selected_vector = {"type": "generic", "description": "ogólny", "rationale": ""}
        else:
            idx = min(vector_index, len(vectors) - 1)
            selected_vector = vectors[idx]

        system_prompt = (
            "Jesteś specjalistą ds. bezpieczeństwa prowadzącym symulację kampanii phishingowej "
            "wyłącznie na potrzeby edukacyjne i badawcze (praca inżynierska). "
            "Generujesz przykładowe treści, które mogą być użyte do szkolenia pracowników "
            "w rozpoznawaniu ataków phishingowych. "
            "Wszystkie linki w treści MUSZĄ mieć postać: [SIMULATED_PHISHING_LINK_FOR_EDUCATION_ONLY]. "
            "ZAWSZE odpowiadaj wyłącznie w formacie JSON."
        )

        user_prompt = (
            "Na podstawie poniższego profilu psychologicznego i wybranego wektora ataku "
            "wygeneruj symulowaną (edukacyjną) wiadomość phishingową.\n\n"
            "PROFIL PSYCHOLOGICZNY:\n"
            f"{profile.get('psychological_profile', '')}\n\n"
            "WYBRANY WEKTOR ATAKU:\n"
            f"{json.dumps(selected_vector, ensure_ascii=False, indent=2)}\n\n"
            "Zwróć wynik WYŁĄCZNIE jako obiekt JSON:\n"
            "{\n"
            '  "subject": "temat e-maila",\n'
            '  "sender_name": "fikcyjna nazwa nadawcy",\n'
            '  "html_body": "treść w HTML (wszystkie linki jako '
            "[SIMULATED_PHISHING_LINK_FOR_EDUCATION_ONLY])\",\n"
            '  "plain_body": "treść plain text (wszystkie linki jako '
            "[SIMULATED_PHISHING_LINK_FOR_EDUCATION_ONLY])\"\n"
            "}\n\n"
            "WAŻNE: To jest symulacja edukacyjna. "
            "Każdy link w treści zastąp dokładnie frazą: [SIMULATED_PHISHING_LINK_FOR_EDUCATION_ONLY]"
        )

        result = self.call_llm(system_prompt, user_prompt)
        # Dodatkowe maskowanie linków jako zabezpieczenie
        result = self._mask_links_in_dict(result)
        return result

    # ------------------------------------------------------------------
    # 4. Maskowanie linków
    # ------------------------------------------------------------------

    # Wyrażenie regularne wykrywające URL-e w tekście
    _URL_PATTERN = re.compile(
        r"https?://[^\s\"'<>\[\]{}|\\^`\x00-\x1f]+"
        r"(?:[^\s\"'<>\[\]{}|\\^`\x00-\x1f,.)]*)",
        re.IGNORECASE,
    )

    def mask_links(self, text: str) -> str:
        """
        Zastępuje wszystkie URL-e w tekście bezpieczną etykietą.

        Args:
            text: Dowolny ciąg znaków mogący zawierać URL-e.

        Returns:
            Tekst z zamaskowanymi linkami.
        """
        return self._URL_PATTERN.sub(SIMULATED_LINK_PLACEHOLDER, text)

    def _mask_links_in_dict(self, data: dict) -> dict:
        """Rekurencyjnie maskuje URL-e we wszystkich wartościach słownika."""
        masked: dict = {}
        for key, value in data.items():
            if isinstance(value, str):
                masked[key] = self.mask_links(value)
            elif isinstance(value, dict):
                masked[key] = self._mask_links_in_dict(value)
            elif isinstance(value, list):
                masked[key] = [
                    self._mask_links_in_dict(item) if isinstance(item, dict)
                    else (self.mask_links(item) if isinstance(item, str) else item)
                    for item in value
                ]
            else:
                masked[key] = value
        return masked

    # ------------------------------------------------------------------
    # 5. Pełny pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        urls: list[str],
        attack_vector_index: int = 0,
    ) -> AnalysisResult:
        """
        Uruchamia pełny 3-etapowy pipeline analizy dla podanych URLi.

        Args:
            urls:                Lista URLi do analizy.
            attack_vector_index: Indeks wektora ataku wybranego do generowania treści.

        Returns:
            AnalysisResult z wynikami wszystkich etapów.
        """
        result = AnalysisResult(urls=urls)

        # --- Scraping ---
        logger.info("=== SCRAPING (%d URLi) ===", len(urls))
        pages = self.scrape_urls(urls)
        successful = [p for p in pages if p.is_ok]

        if not successful:
            logger.error("Nie udało się pobrać żadnej strony. Przerywam pipeline.")
            return result

        combined_text = "\n\n---\n\n".join(
            f"[Źródło: {p.url}]\n{p.text}" for p in successful
        )
        result.raw_texts = [p.text for p in successful]
        logger.info("Pobrano treść z %d/%d stron.", len(successful), len(pages))

        # --- Etap 1 ---
        logger.info("=== ETAP 1: SYNTEZA DANYCH ===")
        result.stage1_facts = self.stage1_synthesize(combined_text)
        logger.info("Fakty: %s", json.dumps(result.stage1_facts, ensure_ascii=False, indent=2))

        if "error" in result.stage1_facts:
            logger.error("Etap 1 zakończony błędem. Przerywam pipeline.")
            return result

        # --- Etap 2 ---
        logger.info("=== ETAP 2: PROFIL BEHAWIORALNY ===")
        result.stage2_profile = self.stage2_profile(result.stage1_facts)
        logger.info(
            "Profil: %s",
            json.dumps(result.stage2_profile, ensure_ascii=False, indent=2),
        )

        if "error" in result.stage2_profile:
            logger.error("Etap 2 zakończony błędem. Przerywam pipeline.")
            return result

        # --- Etap 3 ---
        logger.info("=== ETAP 3: GENEROWANIE TREŚCI ===")
        result.stage3_email = self.stage3_generate(
            result.stage2_profile,
            vector_index=attack_vector_index,
        )
        logger.info(
            "E-mail: %s",
            json.dumps(result.stage3_email, ensure_ascii=False, indent=2),
        )

        logger.info("=== PIPELINE ZAKOŃCZONY ===")
        return result


# ---------------------------------------------------------------------------
# Funkcja pomocnicza: zapis raportu do pliku JSON
# ---------------------------------------------------------------------------

def save_report(result: AnalysisResult, output_path: str = "report.json") -> None:
    """Zapisuje wyniki analizy do pliku JSON."""
    report = {
        "urls": result.urls,
        "stage1_facts": result.stage1_facts,
        "stage2_profile": result.stage2_profile,
        "stage3_email": result.stage3_email,
        "_disclaimer": (
            "Ten raport został wygenerowany wyłącznie na potrzeby edukacyjne "
            "i badawcze w ramach pracy inżynierskiej. "
            "Wszelkie treści są symulowane i nie mogą być użyte do rzeczywistych ataków."
        ),
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info("Raport zapisany do: %s", output_path)


# ---------------------------------------------------------------------------
# Punkt wejścia – dane testowe (mock data)
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Funkcja główna demonstrująca działanie systemu na przykładowych URLach.

    Aby uruchomić:
        1. Uruchom LM Studio i załaduj model (np. Llama-3-8B-Instruct).
        2. Uruchom serwer w LM Studio (Start Server, domyślnie port 1234).
        3. Zmień wartość MODEL_NAME na nazwę załadowanego modelu.
        4. Uruchom: python phishing_research_engine.py

    CEL EDUKACYJNY: Skrypt demonstruje techniki OSINT wyłącznie w celach badawczych.
    """
    # --- Konfiguracja ---
    MODEL_NAME = "local-model"          # Zmień na nazwę modelu w LM Studio
    ATTACK_VECTOR = 0                   # 0 = techniczny, 1 = rekrutacyjny, 2 = hobbystyczny

    # --- Przykładowe publiczne URLe (mock data) ---
    # Możesz zastąpić je dowolnymi publicznymi adresami URL do celów testowych.
    TEST_URLS = [
        "https://github.com/torvalds",                     # profil publiczny GitHub
        "https://en.wikipedia.org/wiki/Linus_Torvalds",   # artykuł Wikipedia
    ]

    # --- Uruchomienie silnika ---
    engine = PhishingResearchEngine(model=MODEL_NAME)
    result = engine.run_pipeline(TEST_URLS, attack_vector_index=ATTACK_VECTOR)

    # --- Wydruk podsumowania ---
    print("\n" + "=" * 60)
    print("PODSUMOWANIE WYNIKÓW (CEL EDUKACYJNY)")
    print("=" * 60)

    print("\n[ETAP 1] Fakty o celu:")
    print(json.dumps(result.stage1_facts, ensure_ascii=False, indent=2))

    print("\n[ETAP 2] Profil behawioralny i wektory ataku:")
    print(json.dumps(result.stage2_profile, ensure_ascii=False, indent=2))

    print("\n[ETAP 3] Wygenerowana (symulowana) wiadomość e-mail:")
    print(json.dumps(result.stage3_email, ensure_ascii=False, indent=2))

    # --- Zapis raportu ---
    save_report(result, "phishing_simulation_report.json")

    print("\nDYSKLAIMER: Powyższe treści są symulowane wyłącznie na potrzeby edukacyjne.")
    print("NIE używaj tego systemu do rzeczywistych ataków.")


if __name__ == "__main__":
    main()
