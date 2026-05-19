class AlphaRegistryError(Exception):
    """Base class for Alpha Registry errors."""


class DuplicateHypothesisError(AlphaRegistryError):
    """Raised when ``(asset, algorithmic_family, hyperparameter_vector)`` already exists.

    The registry must be immutable and de-duplicated to mathematically prove the
    total number of independent trials for the Deflated Sharpe Ratio penalty
    (Bailey & López de Prado 2014).
    """
