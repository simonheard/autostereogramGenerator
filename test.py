import requests
from os.path import basename
from urllib.parse import unquote

def send_get_request(url, params):
    """
    Sends a GET request to the specified URL with the provided query parameters.
    Saves the response content to a file if the Content-Disposition header is present.

    Args:
        url (str): The URL to which the GET request is sent.
        params (dict): Dictionary of query parameters to send in the request.

    Returns:
        Response: The response from the server.
    """
    response = requests.get(url, params=params)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    
    # Check for the Content-Disposition header to save the file
    content_disposition = response.headers.get('Content-Disposition')
    if content_disposition:
        # Extract filename from Content-Disposition header
        filename = unquote(content_disposition.split('filename=')[-1])
        # Save the content to the file
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File saved as {filename}.")
    else:
        print("Response Content:", response.text)
    return response

# URL of the deployed Google Cloud Function
function_url = "https://us-central1-reliable-banner-421414.cloudfunctions.net/process_request"
# function_url = "http://localhost:8080"

# Different sets of parameters to test various scenarios
test_parameters = [
    {"shape": "300,300", "radius": "50", "pattern_shape": "30,30", "levels": "10", "shift_amplitude": "0.5", "invert": "y"}  # Valid custom parameters

]

# Testing each set of parameters
for params in test_parameters:
    print("Testing with parameters:", params)
    send_get_request(function_url, params)
    print("-" * 60)  # Separator for readability
