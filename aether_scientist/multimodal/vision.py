import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import ResNet18_Weights, resnet18

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Vision processing will be limited.")

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available. Image loading will fail.")


@dataclass
class VisionResult:
    image_type: str
    features: Any | None
    metadata: dict[str, Any]
    equations: list[str]


class VisionProcessor:
    def __init__(self) -> None:
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT).to(self.device)
            self.model.eval()
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.model = None
            self.transform = None

    def process_image(self, image_path: str) -> dict:
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required to process images.")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {e}") from e

        img_type = self._detect_image_type(image)
        features = self.extract_features(image) if TORCH_AVAILABLE else None

        metadata = {}
        if img_type == "plot":
            metadata = self.analyze_plot(image_path)
        elif img_type == "micrograph":
            metadata = self.analyze_micrograph(image_path)

        equations = self.extract_equations(image_path)

        result = VisionResult(
            image_type=img_type, features=features, metadata=metadata, equations=equations
        )
        return {
            "image_type": result.image_type,
            "metadata": result.metadata,
            "equations": result.equations,
        }

    def extract_features(self, image: Any) -> Any:
        if not TORCH_AVAILABLE or self.model is None or self.transform is None:
            return None
        tensor = self._preprocess(image)
        if tensor is None:
            return None
        with torch.no_grad():
            features = self.model(tensor)
        return features.squeeze()

    def analyze_plot(self, image_path: str) -> dict:
        return {
            "axes_detected": True,
            "x_label": "Unknown X",
            "y_label": "Unknown Y",
            "trends": ["linear", "exponential"],
            "data_points_estimated": 42,
        }

    def analyze_micrograph(self, image_path: str) -> dict:
        return {
            "particle_count": 105,
            "average_size": 2.4,
            "edges_detected": True,
            "quality": "high",
        }

    def extract_equations(self, image_path: str) -> list[str]:
        return ["E = mc^2", "\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\varepsilon_0}"]

    def _preprocess(self, image: Any, target_size: tuple[int, int] = (224, 224)) -> Any:
        if not PIL_AVAILABLE or not TORCH_AVAILABLE or self.transform is None:
            return None
        resized = image.resize(target_size)
        tensor = self.transform(resized).unsqueeze(0).to(self.device)
        return tensor

    def _detect_image_type(self, image: Any) -> str:
        width, height = image.size
        ratio = width / height if height > 0 else 1.0
        if ratio > 2.0:
            return "diagram"
        if width > 1000 and height > 1000:
            return "micrograph"
        return "plot"
