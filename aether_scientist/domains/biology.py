from aether_scientist.domains.base import DomainAdapter


class BiologyAdapter(DomainAdapter):
    def system_prompt(self) -> str:
        return (
            "You are a research biologist with expertise in genetics, ecology, and cell biology. "
            "Apply evolutionary analysis, systems biology, and statistical genetics. "
            "Reference established biological principles and cite peer-reviewed studies."
        )

    def terminology(self) -> dict[str, str]:
        return {
            "allele": "genetics",
            "phenotype": "genetics",
            "transcription": "genetics",
            "epigenetics": "genetics",
            "trophic level": "ecology",
            "biome": "ecology",
            "symbiosis": "ecology",
            "mitochondria": "cell biology",
            "apoptosis": "cell biology",
            "cytoskeleton": "cell biology",
            "synapse": "neuroscience",
            "action potential": "neuroscience",
            "neurotransmitter": "neuroscience",
            "natural selection": "evolution",
            "speciation": "evolution",
            "phylogeny": "evolution",
            "plasmid": "microbiology",
            "pathogen": "microbiology",
            "biofilm": "microbiology",
            "homeostasis": "systems biology",
            "metabolism": "cell biology",
            "antigen": "immunology",
        }

    def reasoning_patterns(self) -> list[str]:
        return [
            "evolutionary analysis",
            "systems biology",
            "statistical genetics",
            "phylogenetic analysis",
            "population dynamics",
        ]

    def experimental_methods(self) -> list[str]:
        return [
            "polymerase chain reaction (PCR)",
            "CRISPR-Cas9 gene editing",
            "flow cytometry",
            "fluorescence microscopy",
            "next-generation sequencing (NGS)",
            "western blotting",
            "enzyme-linked immunosorbent assay (ELISA)",
            "cell culture",
        ]
