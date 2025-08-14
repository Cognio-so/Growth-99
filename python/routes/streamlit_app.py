# streamlit_app.py

import streamlit as st
import requests

from io import BytesIO  # Keep this for potential future use, though not needed now
import base64          # Keep this for potential future use, though not needed now

# --- Configuration ---
# These URLs must match where your FastAPI applications are running.
SCREENSHOT_API_URL = "http://127.0.0.1:8000/screenshot"
SCRAPE_API_URL = "http://127.0.0.1:8001/scrape-enhanced"

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Web Scraper UI", layout="wide")

st.title("🔎 Web Scraper Test UI")
st.markdown("This app provides a user interface to test the screenshot and content scraping APIs.")

# --- UI Elements ---
api_choice = st.radio(
    "Select the API to test:",
    ("Scrape Content", "Take Screenshot"),
    horizontal=True,
)

url_to_scrape = st.text_input(
    "Enter the URL you want to scrape:",
    "https://firecrawl.dev"
)

submit_button = st.button("🚀 Scrape Now")

st.divider()

# --- Backend Logic ---
if submit_button:
    if not url_to_scrape:
        st.warning("Please enter a URL to scrape.")
    else:
        payload = {"url": url_to_scrape}
        
        if api_choice == "Scrape Content":
            with st.spinner(f"Scraping content from {url_to_scrape}..."):
                try:
                    response = requests.post(SCRAPE_API_URL, json=payload, timeout=60)
                    response.raise_for_status()  # Will raise an exception for 4xx/5xx status codes
                    
                    data = response.json()
                    
                    st.subheader("✅ Scraping Successful!")
                    
                    col1, col2 = st.columns(2)

                    with col1:
                        st.text_area(
                            "Formatted Content (for AI)",
                            data.get("content", ""),
                            height=400
                        )
                    with col2:
                        st.markdown(f"**Title:** {data.get('structured', {}).get('title', 'N/A')}")
                        st.markdown(f"**Description:** {data.get('structured', {}).get('description', 'N/A')}")
                        st.markdown(f"**Cached Result:** `{data.get('metadata', {}).get('cached', 'False')}`")
                    
                    st.subheader("Full Response Metadata")
                    st.json(data.get("metadata", {}))

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to the scraping API. Is it running? Error: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        elif api_choice == "Take Screenshot":
            with st.spinner(f"Capturing screenshot of {url_to_scrape}..."):
                try:
                    response = requests.post(SCREENSHOT_API_URL, json=payload, timeout=60)
                    response.raise_for_status()

                    api_response_data = response.json()
                    
                    # Directly get the URL from the top level of the response
                    image_url = api_response_data.get("screenshot")
                    
                    st.subheader("✅ Screenshot Captured!")
                    
                    if image_url:
                        # Pass the URL directly to st.image()
                        st.image(
                            image_url,
                            caption=f"Screenshot of {url_to_scrape}",
                            use_column_width=True
                        )
                    else:
                        st.error("Could not find the screenshot URL in the API response.")

                    st.subheader("Page Metadata")
                    st.json(api_response_data.get("metadata", {}))

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to the screenshot API. Is it running? Error: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")