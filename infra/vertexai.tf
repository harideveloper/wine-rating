resource "google_vertex_ai_metadata_store" "store" {
  provider    = google-beta
  name        = "default"
  description = "default metadata store"
  project     = var.project_id
}
