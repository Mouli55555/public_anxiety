# test_gemini.py
import os
import google.generativeai as genai

print("--- Starting Gemini Connection Test ---")

try:
    # Step 1: Get the API key from the environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("ERROR: GOOGLE_API_KEY environment variable not found.")
    
    print("API Key found.")
    
    # Step 2: Configure the library
    genai.configure(api_key=api_key)
    print("GenAI library configured.")
    
    # Step 3: Create the model
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Model 'gemini-1.5-flash-latest' initialized.")

    # Step 4: Make the API call
    print("Attempting to generate content...")
    response = model.generate_content("Hello, world.")
    
    # Step 5: Check the response
    print("\n--- ✅ SUCCESS! ---")
    print("Successfully received a response from the Gemini API.")
    print("Response Text:", response.text)

except Exception as e:
    print("\n--- ❌ FAILURE! ---")
    print("The test failed with the following error:")
    print(e)