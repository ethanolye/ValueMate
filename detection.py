from google import generativeai
from flask import Flask, request, jsonify, render_template
from PIL import Image
import os
import base64
import io
import json
import requests
from bs4 import BeautifulSoup
from statistics import median
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure Gemini API only
generativeai.configure(api_key='AIzaSyA1nE9p0aD0I7wBy4hzE5JkDwRdF-X2T2Q')
model = generativeai.GenerativeModel('gemini-2.0-flash')

def search_google_shopping(query):
    """Simple Google Shopping scraper for top 5 results"""
    try:
        search_query = query.replace(' ', '+')
        url = f'https://www.google.com/search?tbm=shop&q={search_query}'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0',
        }
        
        print(f"\nSearching: {url}")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            products = []
            prices = []
            
            # Find all links and prices on the page
            all_links = soup.find_all('a', href=True)
            all_prices = soup.find_all(text=lambda text: text and '$' in text)
            
            # Filter and process product links
            product_links = [
                link for link in all_links 
                if '/shopping/product/' in link.get('href', '')
            ][:5]  # Get top 5 results
            
            print(f"\nFound {len(product_links)} product links")
            
            for link in product_links:
                try:
                    # Get full product URL
                    product_url = 'https://www.google.com' + link['href']
                    
                    # Find closest price element
                    price_elem = link.find_next(text=lambda text: text and '$' in text)
                    if price_elem:
                        price_text = price_elem.strip()
                        # Extract numeric price using regex
                        import re
                        price_match = re.search(r'\$(\d+\.?\d*)', price_text)
                        if price_match:
                            price = float(price_match.group(1))
                            
                            # Get title from link or parent element
                            title = link.get_text().strip()
                            if not title:
                                title = link.find_parent().get_text().strip()
                            
                            products.append({
                                'title': title,
                                'price': price,
                                'link': product_url
                            })
                            prices.append(price)
                            
                            # Print to terminal
                            print(f"\nProduct Found:")
                            print(f"Title: {title}")
                            print(f"Price: ${price:.2f}")
                            print(f"Link: {product_url}")
                            print("-" * 40)
                
                except Exception as e:
                    print(f"Error parsing item: {str(e)}")
                    continue
            
            if products:
                median_price = median(prices)
                print(f"\nMedian Price: ${median_price:.2f}")
                print(f"Total prices found: {len(prices)}")
                return {
                    'products': products,
                    'median_price': median_price
                }
            
            print("No products found")
            return None
            
        print(f"Failed to access URL (Status: {response.status_code})")
        return None
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return None

@app.route('/detect', methods=['POST'])
def detect_object():
    try:
        # Get base64 image data from request
        image_data = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Prompt for Gemini
        prompt = """
        Analyze this image and return ONLY a JSON object with these exact attributes:
        {
            "main_info": {
                "brand": "exact brand name if visible, otherwise leave empty string",
                "colors": "main colors as comma-separated list",
                "category": "specific product category (be very specific)"
            }
        }
        If you cannot determine any attribute with certainty, return an empty string for that attribute.
        Be specific and accurate with the attributes you can determine.
        """
        
        # Generate response from Gemini
        response = model.generate_content([prompt, img])
        result_text = response.text.strip()
        
        # Print raw Gemini response
        print("\n=== Raw Gemini Response ===")
        print("-" * 50)
        print(result_text)
        print("-" * 50)
        
        if result_text.startswith('```json'):
            result_text = result_text[7:-3]
        
        # Parse product info
        product_info = json.loads(result_text)
        
        # Create search query from non-empty values
        query_parts = []
        main_info = product_info['main_info']
        
        if main_info['brand']:
            query_parts.append(main_info['brand'])
        if main_info['category']:
            query_parts.append(main_info['category'])
        if main_info['colors']:
            query_parts.append(main_info['colors'])
            
        search_query = ' '.join(query_parts)
        
        if not search_query.strip():
            return jsonify({'error': 'Not enough identifiable information'}), 400
        
        # Print search details
        print("\n=== Search Details ===")
        print(f"Search Query: {search_query}")
        print("-" * 50)
        
        # Search Google Shopping
        search_results = search_google_shopping(search_query)
        
        if not search_results:
            return jsonify({'error': 'No products found'}), 404
        
        return jsonify({
            'product_info': product_info['main_info'],
            'search_query': search_query,
            'similar_products': search_results['products'],
            'median_price': search_results['median_price']
        })
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print("\n=== Error ===")
        print("-" * 50)
        print(error_msg)
        print("-" * 50)
        return jsonify({'error': error_msg}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)