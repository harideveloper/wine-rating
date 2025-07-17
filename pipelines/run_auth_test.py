from google.auth import default
from google.auth.transport.requests import Request
from google.cloud import storage

def main():
    credentials, project = default()

    credentials.refresh(Request())

    print(f"Authenticated as: {credentials.service_account_email if hasattr(credentials, 'service_account_email') else 'Unknown'}")

    client = storage.Client(credentials=credentials, project=project)
    buckets = list(client.list_buckets())
    print(f"Buckets in project {project}: {[b.name for b in buckets]}")

if __name__ == "__main__":
    main()
