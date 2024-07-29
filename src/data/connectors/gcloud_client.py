import gspread
from dotenv import load_dotenv
import os

load_dotenv()

try:
    gcloud_client = gspread.service_account_from_dict(
        {
            "type": "service_account",
            "project_id": os.environ.get("GCLOUD_SVC_ACCT_PROJECT_ID"),
            "private_key_id": os.environ.get("GCLOUD_SVC_ACCT_PRIVATE_KEY_ID"),
            "private_key": os.environ.get("GCLOUD_SVC_ACCT_PRIVATE_KEY"),
            "client_email": os.environ.get("GCLOUD_SVC_ACCT_CLIENT_EMAIL"),
            "client_id": os.environ.get("GCLOUD_SVC_ACCT_CLIENT_ID"),
            "auth_uri": os.environ.get("GCLOUD_AUTH_URI"),
            "token_uri": os.environ.get("GCLOUD_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.environ.get(
                "GCLOUD_AUTH_PROVIDER_X509_CERT_URL"
            ),
            "client_x509_cert_url": os.environ.get(
                "GCLOUD_SVC_ACCT_CLIENT_X509_CERT_URL"
            ),
            "universe_domain": "googleapis.com",
        }
    )

except Exception:
    gcloud_client = None
