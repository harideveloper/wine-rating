
resource "google_vertex_ai_metadata_store" "store" {
  provider    = google-beta
  name        = "default"
  description = "default metadata store"
  project     = var.project_id

  depends_on = [google_project_service.enable_services]
}

# Create a service account for the Workbench instance
resource "google_service_account" "workbench_sa" {
  account_id   = "workbench-service-account"
  display_name = "Vertex AI Workbench Service Account"
}

# Grant necessary roles to the service account
resource "google_project_iam_member" "workbench_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.workbench_sa.email}"
}

resource "google_project_iam_member" "workbench_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.workbench_sa.email}"
}

# Create Vertex AI Workbench instance
resource "google_workbench_instance" "wine_workbench" {
  name     = "wine-rating-workbench"
  location = "europe-west2-b"

  gce_setup {
    machine_type = "n1-standard-1"
    vm_image {
      project = "cloud-notebooks-managed"
      family  = "workbench-instances"
    }
    service_accounts {
      email = google_service_account.workbench_sa.email
    }
    boot_disk {
      disk_size_gb = 50
      disk_type    = "PD_STANDARD"
    }
  }

  depends_on = [google_project_service.enable_services]
}

# # Create startup script for Workbench instance
# resource "google_storage_bucket_object" "workbench_startup_script" {
#   name    = "workbench-startup-script.sh"
#   bucket  = google_storage_bucket.codebuild.name
#   content = <<-EOT
#     #!/bin/bash

#     # Create project directory
#     mkdir -p /home/jupyter/wine-rating/pipeline

#     # Download code from GCS
#     gsutil cp gs://${google_storage_bucket.codebuild.name}/wine-rating/pipeline/constants.py /home/jupyter/wine-rating/pipeline/
#     gsutil cp gs://${google_storage_bucket.codebuild.name}/wine-rating/pipeline/run_pipeline.py /home/jupyter/wine-rating/pipeline/
#     gsutil cp gs://${google_storage_bucket.codebuild.name}/wine-rating/pipeline/wine_rating_pipeline.py /home/jupyter/wine-rating/pipeline/
#     gsutil cp gs://${google_storage_bucket.codebuild.name}/wine-rating/requirements.txt /home/jupyter/wine-rating/
#     gsutil cp gs://${google_storage_bucket.codebuild.name}/wine-rating/Makefile /home/jupyter/wine-rating/

#     # Set correct permissions
#     chown -R jupyter:jupyter /home/jupyter/wine-rating

#     # Create virtual environment and install dependencies
#     su - jupyter -c "cd /home/jupyter/wine-rating && python -m venv wine_venv && wine_venv/bin/pip install -U pip && wine_venv/bin/pip install -r requirements.txt"
#   EOT
# }


# # Output the Workbench URL
# output "workbench_url" {
#   value       = "https://${google_notebooks_instance.wine_workbench.name}.${var.region}.notebooks.googleapis.com/lab"
#   description = "URL to access the Vertex AI Workbench instance"
# }

# output "workbench_setup_instructions" {
#   value = <<-EOT
#     Your Vertex AI Workbench instance has been created!

#     To run the wine rating pipeline:
#     1. Go to the Workbench URL: ${google_notebooks_instance.wine_workbench.name}.${var.region}.notebooks.googleapis.com
#     2. Open a terminal in the Workbench
#     3. Navigate to the wine-rating directory: cd /home/jupyter/wine-rating
#     4. Run the pipeline: cd pipeline && ../wine_venv/bin/python run_pipeline.py

#     The environment is already set up with all the required dependencies.
#   EOT
# }
