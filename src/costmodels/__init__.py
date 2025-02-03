import os  # fmt:skip
os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"

from costmodels.dtu_offshore import (
    DTUOffshoreCM,
    DTUOffshoreCMInput,
    DTUOffshoreCMOutput,
)
