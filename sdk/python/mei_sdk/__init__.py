"""MEI Runtime Python SDK (experimental).

Public names are MEI Runtime / mei-1.0-51m. This package does not expose needle2.
"""

from .engine import Engine, Session, load_model
from .cq2 import CqTensor, TensorContainer, TensorToPack, build_tensor_container
from .errors import SdkError
from .package import HeadReport, ModelPackage, load_package
from .protocol import adapt_v1_request, normalize_request, parse_v2_text, render_request, schema_fingerprint
from .shared import UnsupportedSchemaError
from .version import sdk_versions

__all__ = [
    "Engine",
    "CqTensor",
    "HeadReport",
    "ModelPackage",
    "SdkError",
    "Session",
    "TensorContainer",
    "TensorToPack",
    "UnsupportedSchemaError",
    "adapt_v1_request",
    "build_tensor_container",
    "load_package",
    "load_model",
    "normalize_request",
    "parse_v2_text",
    "render_request",
    "schema_fingerprint",
    "sdk_versions",
]
