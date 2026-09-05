import importlib

from aether_scientist.domains.base import DomainAdapter

ADAPTER_REGISTRY: dict[str, str] = {
    "physics": "aether_scientist.domains.physics.PhysicsAdapter",
    "chemistry": "aether_scientist.domains.chemistry.ChemistryAdapter",
    "biology": "aether_scientist.domains.biology.BiologyAdapter",
}


def load_adapter(domain: str) -> DomainAdapter:
    if domain not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown domain: {domain}")

    module_path, class_name = ADAPTER_REGISTRY[domain].rsplit(".", 1)
    module = importlib.import_module(module_path)
    adapter_class = getattr(module, class_name)
    return adapter_class()


def available_domains() -> list[str]:
    return list(ADAPTER_REGISTRY.keys())
