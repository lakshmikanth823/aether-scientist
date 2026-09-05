from aether_scientist.domains.base import DomainAdapter


class PhysicsAdapter(DomainAdapter):
    def system_prompt(self) -> str:
        return (
            "You are a theoretical and experimental physicist. "
            "Apply conservation laws, symmetry analysis, and dimensional analysis. "
            "Use precise mathematical reasoning. Reference established physical laws "
            "and cite relevant experiments when applicable."
        )

    def terminology(self) -> dict[str, str]:
        return {
            "wavefunction": "quantum mechanics",
            "entanglement": "quantum mechanics",
            "superposition": "quantum mechanics",
            "hamiltonian": "quantum mechanics",
            "spacetime": "relativity",
            "geodesic": "relativity",
            "lorentz factor": "relativity",
            "entropy": "thermodynamics",
            "enthalpy": "thermodynamics",
            "partition function": "thermodynamics",
            "maxwell equations": "electromagnetism",
            "magnetic flux": "electromagnetism",
            "permittivity": "electromagnetism",
            "lagrangian": "mechanics",
            "moment of inertia": "mechanics",
            "coriolis force": "mechanics",
            "diffraction": "optics",
            "refractive index": "optics",
            "interference": "optics",
            "photon": "quantum mechanics",
            "boson": "particle physics",
            "fermion": "particle physics",
        }

    def reasoning_patterns(self) -> list[str]:
        return [
            "conservation laws",
            "symmetry analysis",
            "dimensional analysis",
            "scale analysis",
            "perturbation theory",
            "variational methods",
        ]

    def experimental_methods(self) -> list[str]:
        return [
            "particle accelerators",
            "interferometry",
            "spectroscopy",
            "calorimetry",
            "electron microscopy",
            "x-ray diffraction",
        ]

    def validate_units(self, expression: str) -> bool:
        """Optional physics-specific unit validation."""
        valid_units = {"kg", "m", "s", "A", "K", "mol", "cd"}
        return any(unit in expression for unit in valid_units)
