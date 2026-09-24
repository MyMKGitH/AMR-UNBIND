class AMRUnbindError(RuntimeError):
    """Base class for expected AMR-UNBIND failures."""


class InputValidationError(AMRUnbindError):
    pass


class ProteinPreparationError(AMRUnbindError):
    pass


class LigandPreparationError(AMRUnbindError):
    pass


class DockingError(AMRUnbindError):
    pass


class SystemConstructionError(AMRUnbindError):
    pass


class SimulationError(AMRUnbindError):
    pass


class ChimeraXError(AMRUnbindError):
    pass
