variable "project_id" {
  description = "Google Cloud Project ID of the File Upload Project"
  type        = string
}

variable "project_number" {
  description = "GCP project number"
  type        = string
}

variable "region" {
  description = "Google Cloud Project ID of the File Upload Project"
  type        = string
  default     = "europe-west2"
}

variable "application" {
  description = "Application or workload prefix"
  type        = string
  default     = "dfc"
}
