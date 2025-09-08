from huggingface_hub import upload_file
from huggingface_hub import HfApi

## create directory if it does not exist
#api = HfApi()
#api.create_repo("mohsensadr91/Food101", private=True)  # private if you want

# Upload weights
upload_file(
    path_or_fileobj="ddpm_Food101.pt",
    path_in_repo="ddpm_Food101.pt",
    repo_id="mohsensadr91/Food101",
    commit_message="Upload DDPM model weights of Food101"
)
