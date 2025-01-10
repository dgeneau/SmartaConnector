
# Install the necessary R packages
# smartabaseR is required for fetching data from Smartabase
if (!requireNamespace("smartabaseR", quietly = TRUE)) {
  install.packages("smartabaseR")
}

# Additional dependencies (if any are needed by smartabaseR)
if (!requireNamespace("httr", quietly = TRUE)) {
  install.packages("httr")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite")
}

# Ensure base R functions are available (these are pre-installed with R)
message("Base R packages are pre-installed with R.")

# Optional: Check for updates for installed packages
update.packages(ask = FALSE)
