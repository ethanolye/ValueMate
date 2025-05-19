from google import generativeai
from flask import Flask, request, jsonify, render_template
from PIL import Image
import os
import base64
import io
import json
import re
import requests
from bs4 import BeautifulSoup
from statistics import median, mean
from dotenv import load_dotenv
from flask_cors import CORS
from numpy import percentile
import numpy as np

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Gemini API only
generativeai.configure(api_key='AIzaSyA1nE9p0aD0I7wBy4hzE5JkDwRdF-X2T2Q')
model = generativeai.GenerativeModel('gemini-2.0-flash')

def get_product_title_from_gemini(html_content):
    """Use Gemini to extract the main product title from HTML content"""
    try:
        prompt = """
        From this HTML content, extract ONLY the main product title/heading.
        Return just the title text, nothing else.
        If you can't find a clear title, return an empty string.
        """
        
        response = model.generate_content([prompt, html_content])
        title = response.text.strip()
        print(f"Gemini extracted title: {title}")
        return title
    except Exception as e:
        print(f"Error getting title from Gemini: {str(e)}")
        return ""

# Add this new function after get_product_title_from_gemini
def get_validated_price_from_gemini(html_content, title):
    """Use Gemini to find and validate price from HTML content"""
    try:
        prompt = f"""
        For the product "{title}", analyze this HTML and extract the main product price.
        You must respond with ONLY a valid JSON object in this exact format, nothing else:
        {{
            "price": "00.00",
            "confidence": "high",
            "explanation": "brief explanation"
        }}
        Rules:
        - price must be numbers only, no currency symbols
        - confidence must be exactly "high", "medium", or "low"
        - ignore shipping costs and related product prices
        - if no clear price is found, use "0.00" as price and "low" as confidence
        """
        
        response = model.generate_content([prompt, html_content])
        response_text = response.text.strip()
        
        # Clean up response if it contains markdown code blocks
        if response_text.startswith('```json'):
            response_text = response_text[7:-3]
        elif response_text.startswith('```'):
            response_text = response_text[3:-3]
            
        # Try to parse JSON
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse Gemini response as JSON: {response_text}")
            return {
                "price": "0.00",
                "confidence": "low",
                "explanation": "Failed to parse price from HTML"
            }
        
        print(f"\nGemini price validation:")
        print(f"Price found: ${result['price']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Explanation: {result['explanation']}")
        
        return result
        
    except Exception as e:
        print(f"Error getting price from Gemini: {str(e)}")
        return {
            "price": "0.00",
            "confidence": "low",
            "explanation": f"Error: {str(e)}"
        }

def remove_outliers(prices):
    """Remove outliers using IQR method"""
    if not prices:
        return []
    
    q1 = np.percentile(prices, 25)
    q3 = np.percentile(prices, 75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    return [x for x in prices if lower_bound <= x <= upper_bound]

def search_google_shopping(query):
    """Simple Google Shopping scraper for top 6 results"""
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
            raw_prices = []
            median_price = 0
            average_price = 0
            pricelist = [] # Initialize pricelist

            # Find all links first
            all_links = soup.find_all('a')
            
            # Filter and process product links
            product_links = [
                link for link in all_links 
                if '/shopping/product/' in link.get('href', '')
              ][:6]  # Increase limit to 6 products
            
            print(f"\nFound {len(product_links)} product links")
            
            # Process all product links instead of just the first one
            for link in product_links:
                try:
                    product_url = 'https://www.google.com' + link['href']
                    print(f"Processing product link: {product_url}")

                    # Fetch the product page
                    product_response = requests.get(product_url)
                    if (product_response.status_code == 200):
                        product_soup = BeautifulSoup(product_response.content, 'html.parser')

                        # Find the 'related' section and exclude it
                        related_section = product_soup.find(id='related')
                        if related_section:
                            for element in related_section.find_all(recursive=True):
                                element.decompose()

                        # Get title using Gemini
                        html_content = str(product_soup)
                        title = get_product_title_from_gemini(html_content)
                        
                        # Fallback to previous method if Gemini returns empty string
                        if not title:
                            title = link.get_text().strip()
                            if not title:
                                title = link.find_parent().get_text().strip()

                        # Get price using Gemini
                        price_validation = get_validated_price_from_gemini(html_content, title)
                        
                        if price_validation:
                            try:
                                price = float(price_validation['price'])
                                if price_validation['confidence'] in ['high', 'medium']:
                                    raw_prices.append(price)
                                    pricelist.append(price)
                                    print(f"Found validated price: ${price:.2f}")
                                else:
                                    print("Low confidence in price, searching for alternatives...")
                                    # Fallback to finding prices with highest frequency
                                    price_elements = product_soup.find_all(
                                        text=re.compile(r'\$\s*[\d\.]+'))
                                    price_counts = {}
                                    for elem in price_elements:
                                        try:
                                            price = float(re.search(r'\$\s*([\d\.]+)', 
                                                                  elem.strip()).group(1))
                                            price_counts[price] = price_counts.get(price, 0) + 1
                                        except (ValueError, AttributeError):
                                            continue
                                    
                                    if price_counts:
                                        most_common_price = max(price_counts.items(), 
                                                             key=lambda x: x[1])[0]
                                        raw_prices.append(most_common_price)
                                        pricelist.append(most_common_price)
                                        print(f"Found alternative price: ${most_common_price:.2f}")
                            
                            except (ValueError, KeyError) as e:
                                print(f"Error processing price: {str(e)}")
                                continue

                        products.append({
                            'title': title,
                            'link': product_url,
                            'average_price': price if price_validation else None
                        })

                        print(f"\nProduct Found:")
                        print(f"Title: {title}")
                        print(f"Link: {product_url}")
                        print("-" * 40)
                    else:
                        print(f"Failed to access product URL (Status: {product_response.status_code})")
                
                except Exception as e:
                    print(f"Error parsing item: {str(e)}")
                    continue

            # Calculate overall statistics after processing all products
            if pricelist:
                # Remove outliers before calculating average
                cleaned_prices = remove_outliers(pricelist)
                average_price = mean(cleaned_prices) if cleaned_prices else None
                median_price = median(cleaned_prices) if cleaned_prices else None
            else:
                average_price = None
                median_price = None

            if products:
                return {
                    'products': products,
                    'median_price': median_price,
                    'average_price': average_price,
                    'total_prices_found': len(raw_prices),
                    'all_raw_prices': raw_prices,
                    'pricelist': pricelist
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
        
        # Store debug data for the debug view
        app.debug_data = {
            'gemini_response': result_text,
            'product_info': product_info['main_info'],
            'search_query': search_query,
            'median_price': search_results['median_price'],
            'average_price': search_results['average_price'],
            'total_prices_found': search_results['total_prices_found'],
            'all_raw_prices': search_results['all_raw_prices'],
            'similar_products': search_results['products'],
            'pricelist': search_results['pricelist']
        }

        return jsonify({
            'product_info': product_info['main_info'],
            'search_query': search_query,
            'similar_products': search_results['products'],
            'median_price': search_results['median_price'],
            'average_price': search_results['average_price'],
            'total_prices_found': search_results['total_prices_found'],
            'all_raw_prices': search_results['all_raw_prices'],
            'pricelist': search_results['pricelist']
        })
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print("\n=== Error ===")
        print("-" * 50)
        print(error_msg)
        print("-" * 50)
        return jsonify({'error': error_msg}), 500

def extract_prices_from_html(html):
    """Extracts all dollar prices from an HTML string."""
    prices = re.findall(r'\$\s*[\d\.]+', html)
    return prices

@app.route('/debug')
def debug():
    similar_products = getattr(app, 'debug_data', {}).get('similar_products', [])
    
    for product in similar_products:
        product['individual_prices'] = extract_prices_from_html(product['raw_html'])

    debug_data = {
        'gemini_response': getattr(app, 'debug_data', {}).get('gemini_response', 'No detection performed yet'),
        'product_info': getattr(app, 'debug_data', {}).get('product_info', {'brand': '', 'colors': '', 'category': ''}),
        'search_query': getattr(app, 'debug_data', {}).get('search_query', ''),
        'median_price': getattr(app, 'debug_data', {}).get('median_price', 0),
        'average_price': getattr(app, 'debug_data', {}).get('average_price', 0),
        'total_prices_found': getattr(app, 'debug_data', {}).get('total_prices_found', 0),
        'all_raw_prices': getattr(app, 'debug_data', {}).get('all_raw_prices', []),
        'similar_products': similar_products,
        'pricelist': getattr(app, 'debug_data', {}).get('pricelist', [])
    }

    return render_template('debug.html', debug_data=debug_data)

@app.route('/')
def index():
    return render_template('index.html')

# Add this new route after the existing routes
@app.route('/update-product', methods=['POST'])
def update_product():
    try:
        data = request.json
        updated_info = {
            'main_info': {
                'brand': data.get('brand', ''),
                'colors': data.get('colors', ''),
                'category': data.get('category', ''),
                'description': data.get('description', '')
            }
        }
        
        # Create new search query from updated info
        query_parts = []
        if updated_info['main_info']['brand']:
            query_parts.append(updated_info['main_info']['brand'])
        if updated_info['main_info']['category']:
            query_parts.append(updated_info['main_info']['category'])
        if updated_info['main_info']['description']:
            query_parts.append(updated_info['main_info']['description'])
            
        search_query = ' '.join(query_parts)
        
        if not search_query.strip():
            return jsonify({'error': 'Not enough information provided'}), 400
            
        # Search Google Shopping with updated info
        search_results = search_google_shopping(search_query)
        
        if not search_results:
            return jsonify({'error': 'No products found'}), 404
            
        return jsonify({
            'product_info': updated_info['main_info'],
            'search_query': search_query,
            'similar_products': search_results['products'],
            'median_price': search_results['median_price'],
            'average_price': search_results['average_price'],
            'total_prices_found': search_results['total_prices_found'],
            'all_raw_prices': search_results['all_raw_prices'],
            'pricelist': search_results['pricelist']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)