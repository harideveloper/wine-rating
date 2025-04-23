resource "google_project_service" "enable_services" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "compute.googleapis.com",
    "notebooks.googleapis.com"
  ])
  service = each.key
}
