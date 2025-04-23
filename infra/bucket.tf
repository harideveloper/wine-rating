resource "google_storage_bucket" "modelbuild" {
  name                        = "model-build-${var.application}-${var.project_id}"
  location                    = var.region
  force_destroy               = true
  storage_class               = "REGIONAL"
  uniform_bucket_level_access = true

  depends_on = [google_project_service.enable_services]
}


# Create a GCS bucket for code storage
resource "google_storage_bucket" "code_bucket" {
  name          = "${var.project_id}-workbench-code"
  location      = var.region
  force_destroy = true
  depends_on    = [google_project_service.storage_api]
}

