#!/bin/bash

# Wine Recommendation System - Environment Setup Script
# This script sets up the required environment for development and deployment

set -e  # Exit on any error

# Parse command line arguments
CREATE_VENV=true
INSTALL_DEPS=true
SETUP_GCP=false
PROJECT_ID=""

print_usage() {
  echo "Usage: $0 [--no-venv] [--no-deps] [--setup-gcp --project-id=<project-id>]"
  echo ""
  echo "Options:"
  echo "  --no-venv         Skip virtual environment creation"
  echo "  --no-deps         Skip dependency installation"
  echo "  --setup-gcp       Setup GCP environment (requires --project-id)"
  echo "  --project-id      GCP project ID (required if --setup-gcp is used)"
}

for i in "$@"; do
  case $i in
    --no-venv)
      CREATE_VENV=false
      shift
      ;;
    --no-deps)
      INSTALL_DEPS=false
      shift
      ;;
    --setup-gcp)
      SETUP_GCP=true
      shift
      ;;