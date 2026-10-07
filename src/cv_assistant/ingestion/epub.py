"""Extraction du texte d'un EPUB.

Un EPUB est une archive zip :
- META-INF/container.xml indique où se trouve le fichier OPF ;
- le fichier OPF liste les fichiers du livre (manifest) et leur ordre de lecture (spine) ;
- chaque chapitre est un fichier XHTML.
"""

from __future__ import annotations

import posixpath
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

_CONTAINER_PATH = "META-INF/container.xml"
_CONTAINER_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:container"
_OPF_NAMESPACES = {"opf": "http://www.idpf.org/2007/opf"}

# Balises dont le contenu n'est pas du texte à lire.
_SKIPPED_HTML_TAGS = frozenset({"head", "script", "style"})
# Balises qui ne séparent pas les mots : "<em>para</em>graphe" doit rester "paragraphe".
_INLINE_HTML_TAGS = frozenset(
    {"a", "abbr", "b", "cite", "code", "em", "i", "s", "small", "span", "strong", "sub", "sup", "u"}
)


class EpubReadError(Exception):
    pass


def read_epub_sections(epub_path: Path) -> list[str]:
    """Renvoie le texte de chaque chapitre de l'EPUB, dans l'ordre de lecture."""
    try:
        with zipfile.ZipFile(epub_path) as archive:
            opf_path = _find_opf_path(archive)
            return [
                _html_to_text(archive.read(chapter_path).decode("utf-8", errors="replace"))
                for chapter_path in _find_chapter_paths(archive, opf_path)
            ]
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as error:
        raise EpubReadError(str(error)) from error


def _find_opf_path(archive: zipfile.ZipFile) -> str:
    """Chemin, dans l'archive, du fichier OPF qui décrit le contenu du livre."""
    container = ElementTree.fromstring(archive.read(_CONTAINER_PATH))
    rootfile = container.find(f".//{{{_CONTAINER_NAMESPACE}}}rootfile")
    opf_path = rootfile.get("full-path") if rootfile is not None else None
    if not opf_path:
        raise EpubReadError(f"fichier OPF introuvable dans {_CONTAINER_PATH}")
    return opf_path


def _find_chapter_paths(archive: zipfile.ZipFile, opf_path: str) -> list[str]:
    """Chemins des chapitres dans l'archive, dans l'ordre de lecture du livre."""
    package = ElementTree.fromstring(archive.read(opf_path))
    href_by_id = {
        item.get("id"): item.get("href")
        for item in package.findall("opf:manifest/opf:item", _OPF_NAMESPACES)
    }
    # Les href du manifest sont relatifs au dossier du fichier OPF.
    opf_dir = posixpath.dirname(opf_path)

    chapter_paths: list[str] = []
    for itemref in package.findall("opf:spine/opf:itemref", _OPF_NAMESPACES):
        href = href_by_id.get(itemref.get("idref"))
        if href:
            chapter_paths.append(posixpath.normpath(posixpath.join(opf_dir, unquote(href))))
    return chapter_paths


def _html_to_text(html: str) -> str:
    extractor = _HtmlTextExtractor()
    extractor.feed(html)
    extractor.close()
    return "".join(extractor.parts)


class _HtmlTextExtractor(HTMLParser):
    """Garde le texte visible et met un espace à la place des balises de bloc."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skipped_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_HTML_TAGS:
            self._skipped_depth += 1
        elif tag not in _INLINE_HTML_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_HTML_TAGS:
            self._skipped_depth = max(0, self._skipped_depth - 1)
        elif tag not in _INLINE_HTML_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skipped_depth:
            self.parts.append(data)
