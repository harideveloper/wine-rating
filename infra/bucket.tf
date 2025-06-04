resource "google_storage_bucket" "modelbuild" {
  name                        = "model-build-${var.application}-${var.project_id}"
  location                    = var.region
  force_destroy               = true
  storage_class               = "REGIONAL"
  uniform_bucket_level_access = true

  depends_on = [google_project_service.enable_services]
}

resource "google_storage_bucket" "codebuild" {
  name                        = "workbench-code-${var.application}-${var.project_id}"
  location                    = var.region
  force_destroy               = true
  storage_class               = "REGIONAL"
  uniform_bucket_level_access = true

  depends_on = [google_project_service.enable_services]
}
