"""
B111 boundary for Bernese RNXSMT runtime parameters.

The supplied Bernese source set contains RNXSMT.f but does not contain
the implementation of RSMTIN (module s_rsmtin).

RNXSMT obtains the scientific screening/arc parameters below from
RSMTIN.  B111 therefore MUST NOT synthesize scientific defaults for
them.
"""

from dataclasses import dataclass


class MissingRsmtinParametersError(RuntimeError):
    """Required Bernese RNXSMT parameters were not supplied."""


@dataclass(frozen=True)
class RnxSmtRuntimeParameters:
    gaparc: float
    gapl4: float
    rmsl5: float
    rmsl4: float
    rmsl3: float
    slpmin: float
    slpout: float
    minobs: int
    minl4: int
    ifxslp: int


def require_rsmtin_parameters(
    parameters: RnxSmtRuntimeParameters | None,
) -> RnxSmtRuntimeParameters:
    """
    Require explicit RNXSMT scientific parameters.

    No Bernese defaults are inferred because RSMTIN is absent from
    the supplied source set.
    """

    if parameters is None:
        raise MissingRsmtinParametersError(
            "B111 requires explicit RNXSMT runtime parameters. "
            "The supplied Bernese sources show these values being "
            "provided by RSMTIN, whose implementation is unavailable. "
            "No scientific defaults will be synthesized."
        )

    return parameters
