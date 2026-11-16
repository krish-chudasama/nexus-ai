"""Domain exceptions. The API maps these to HTTP status codes."""

from __future__ import annotations


class NexusError(Exception):
    status_code = 500
    code = "internal_error"


class ConfigurationError(NexusError):
    status_code = 503
    code = "configuration_error"


class ValidationFailed(NexusError):
    status_code = 422
    code = "validation_error"


class UnsupportedFileType(ValidationFailed):
    code = "unsupported_file_type"


class FileTooLarge(NexusError):
    status_code = 413
    code = "file_too_large"


class DuplicateDocument(NexusError):
    status_code = 409
    code = "duplicate_document"


class ResourceNotFound(NexusError):
    status_code = 404
    code = "not_found"


class DocumentNotFound(ResourceNotFound):
    code = "document_not_found"


class ExtractionError(NexusError):
    status_code = 422
    code = "extraction_failed"


class LLMError(NexusError):
    status_code = 502
    code = "llm_error"


class ToolError(NexusError):
    status_code = 400
    code = "tool_error"
