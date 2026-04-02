import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from correct absolute path
env_path = Path("/Users/harshit/Documents/WEBSITE_EXPLO/.env.local")
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("TWOSLIDES_API_KEY") or "sk-2slides-03e19fc6af676591a9a362572863d859bf0f0b75e51c4c09551a7ce1ce046f90"
BASE_URL = "https://api.2slides.com/api/v1"

def get_headers():
    if not API_KEY:
        print("Error: TWOSLIDES_API_KEY is not set in .env.local")
        exit(1)
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "x-api-key": API_KEY, # Supplying both as API standard varies
    }

def read_content():
    """Reads the presentation prompt and technical documentation to feed to the AI."""
    content = ""
    guide_path = Path("/Users/harshit/.gemini/antigravity/brain/9e406945-96d6-4a0f-8435-5c895b377489/artifacts/Google_Slides_Prompt_Guide.md")
    tech_path = Path("/Users/harshit/.gemini/antigravity/brain/9e406945-96d6-4a0f-8435-5c895b377489/artifacts/SignBridge_Technical_Documentation.md")
    
    if guide_path.exists():
        content += guide_path.read_text() + "\n\n"
    if tech_path.exists():
        content += tech_path.read_text()
        
    if not content:
        print("Error: Could not find Google_Slides_Prompt_Guide.md or SignBridge_Technical_Documentation.md")
        exit(1)
        
    return content

def generate_slides(text_content):
    print("Initiating presentation generation via 2slides API (Nano Banana Engine)...")
    url = f"{BASE_URL}/slides/create-pdf-slides"
    
    # We want to format the design style based on our SignBridge website design system Let's ensure modern dark theme 
    payload = {
        "userInput": text_content,
        "designStyle": "modern, dark background, bold typography, indigo and violet neon colors, tech aesthetic",
        "aspectRatio": "16:9",
        "resolution": "2K",
        "page": 17, # As per our 17 slide prompt guide
        "contentDetail": "standard",
        "mode": "async"
    }

    try:
        response = requests.post(url, json=payload, headers=get_headers())
        response.raise_for_status()
        data = response.json()
        job_id = data.get("jobId")
        if not job_id:
             print("Successful response but no jobId returned. Sync URL might be available:", data)
             return data.get("downloadUrl")
             
        print(f"Job started successfully! Job ID: {job_id}")
        return poll_job(job_id)

    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
        if e.response is not None:
            print("Response:", e.response.text)
        exit(1)

def poll_job(job_id):
    print("Polling for completion. This can take a few minutes for Nano Banana jobs...")
    url = f"{BASE_URL}/jobs/{job_id}"
    
    while True:
        try:
            response = requests.get(url, headers=get_headers())
            response.raise_for_status()
            data = response.json()
            
            status = data.get("status", "pending").lower()
            
            if status == "success":
                download_url = data.get("downloadUrl")
                print("\n✅ Presentation generated successfully!")
                return download_url
            elif status == "failed":
                print(f"\n❌ Job failed. Message: {data.get('message', 'Unknown error')}")
                exit(1)
            else:
                print(f"Status: {status}... waiting 15 seconds.")
                time.sleep(15)
                
        except requests.exceptions.RequestException as e:
            print(f"Status check failed: {e}")
            time.sleep(15) # Wait and try again

def download_file(url, output_filename="SignBridge_Presentation.pptx"):
    print(f"Downloading your presentation to {output_filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✨ Download complete! File saved as {output_filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download the presentation: {e}")
        print(f"You can manually download it here: {url}")

if __name__ == "__main__":
    content = read_content()
    # To avoid payload limit size, we might want to truncate extremely large content. 
    # The Prompt guide shouldn't be larger than a few KB, so it's safe.
    print(f"Read {len(content)} characters of presentation content.")
    
    download_url = generate_slides(content)
    if download_url:
        download_file(download_url)
