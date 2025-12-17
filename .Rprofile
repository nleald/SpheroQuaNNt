# Configure Python environment for ShinyApps deployment
library(reticulate)
python_env <- "/home/shiny/.virtualenvs/cnn_env"
if (nzchar(Sys.getenv("SHINY_PORT"))) {
  Sys.setenv(RETICULATE_PYTHON = "/usr/bin/python3")
} else {
  if (!dir.exists(python_env)) {
    virtualenv_create(envname = python_env, python = Sys.which("python3"))
  }
  use_virtualenv(python_env, required = TRUE)
  Sys.setenv(RETICULATE_PYTHON = file.path(python_env, "bin/python"))
}

# Increase timeout for rsconnect deployment (useful for large bundles)
options(rsconnect.http.timeout = 300)

# Ensure this file ends with a newline
