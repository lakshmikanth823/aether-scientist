from aether_scientist.domains.base import DomainAdapter


class ChemistryAdapter(DomainAdapter):
    def system_prompt(self) -> str:
        return (
            "You are an expert chemist specializing in organic, inorganic, and physical chemistry. "
            "Analyze reaction mechanisms, thermodynamic feasibility, and kinetic pathways. "
            "Reference IUPAC nomenclature and cite relevant literature."
        )

    def terminology(self) -> dict[str, str]:
        return {
            "nucleophile": "organic chemistry",
            "electrophile": "organic chemistry",
            "chirality": "organic chemistry",
            "stereoisomer": "organic chemistry",
            "ligand": "inorganic chemistry",
            "coordination number": "inorganic chemistry",
            "oxidation state": "inorganic chemistry",
            "enzyme": "biochemistry",
            "peptide bond": "biochemistry",
            "metabolite": "biochemistry",
            "titrant": "analytical chemistry",
            "analyte": "analytical chemistry",
            "calibration curve": "analytical chemistry",
            "activation energy": "physical chemistry",
            "gibbs free energy": "physical chemistry",
            "chemical potential": "physical chemistry",
            "catalysis": "physical chemistry",
            "polymerization": "organic chemistry",
            "hybridization": "physical chemistry",
            "buffer": "analytical chemistry",
            "solubility product": "physical chemistry",
            "allotrope": "inorganic chemistry",
        }

    def reasoning_patterns(self) -> list[str]:
        return [
            "reaction mechanisms",
            "thermodynamic analysis",
            "kinetic modeling",
            "spectral interpretation",
            "stoichiometry",
        ]

    def experimental_methods(self) -> list[str]:
        return [
            "nuclear magnetic resonance (NMR)",
            "mass spectrometry (MS)",
            "high-performance liquid chromatography (HPLC)",
            "gas chromatography (GC)",
            "titration",
            "x-ray crystallography",
            "infrared spectroscopy (IR)",
            "ultraviolet-visible spectroscopy (UV-Vis)",
        ]
