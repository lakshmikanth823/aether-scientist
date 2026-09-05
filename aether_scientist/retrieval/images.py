import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ImageAsset:
    """Represents an extracted image asset from a document."""

    doc_id: str
    page_num: int
    image_path: str
    bbox: tuple[float, float, float, float] | None = None


def extract_images(
    pdf_path: str | Path, cache_dir: str | Path = ".aether_cache"
) -> list[ImageAsset]:
    """Extract embedded images from PDF pages and save them to disk."""
    p = Path(pdf_path)
    if not p.exists() or p.suffix.lower() != ".pdf":
        return []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed; skipping PDF image extraction.")
        return []

    doc_id = hashlib.sha256(f"{p.name}:{p.stat().st_size}".encode()).hexdigest()[:12]
    out_dir = Path(cache_dir) / "images" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    assets: list[ImageAsset] = []
    try:
        doc = fitz.open(p)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                if not base_img:
                    continue
                ext = base_img.get("ext", "png")
                img_bytes = base_img.get("image", b"")
                if not img_bytes:
                    continue
                img_file = out_dir / f"img_p{page_idx + 1}_{img_idx + 1}.{ext}"
                img_file.write_bytes(img_bytes)
                assets.append(
                    ImageAsset(
                        doc_id=doc_id,
                        page_num=page_idx + 1,
                        image_path=str(img_file),
                    )
                )
        doc.close()
    except Exception as e:
        logger.warning(f"Error extracting images from {p}: {e}")
        return []

    return assets
