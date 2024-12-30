import json
import os
import streamlit as st
import datetime
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher
import random

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
intents = [
    {
        "tag": "greeting",
        "patterns":
            ["Hi", "Hello", "Hey", "Hi there", "Hello there", "Hi how are you?", "What's up?", "Hey what's up?"],
        "responses":
            ["Hello! Welcome to our online store. How can I assist you today?",
             "Hi there! How can I help you with your shopping needs?", "Hey! What brings you to our website today?"]
    },
    {
        "tag": "product_inquiry",
        "patterns": [
            "What products do you offer?",
            "Can you show me your products?",
            "I'm looking for a product",
            "Do you have [product name]?",
            "Can I get information about [product name]?"
        ],
        "responses": [
            "We offer a wide range of products. Please visit our website to explore our collection.",
            "Our product catalog is available on our website. Would you like me to guide you through it?",
            "Please let me know what product you're looking for, and I'll do my best to assist you.",
            "Yes, we do have [product name]. Would you like to know more about it?",
            "I'd be happy to provide information about [product name]. Please let me know what you'd like to know."
        ]
    },
    {
        "tag": "product_recommendation",
        "patterns": [
            "I'm looking for a gift",
            "Can you recommend a product?",
            "What's the best product for [occasion]?",
            "I need a product for [purpose]",
            "Can you suggest a product based on my preferences?"
        ],
        "responses": [
            "We'd be happy to help you find a gift. Can you please provide more information about the recipient's interests?",
            "Please let me know what type of product you're looking for, and I'll do my best to recommend something.",
            "Our best-selling product for [occasion] is [product name]. Would you like to know more about it?",
            "I'd be happy to recommend a product for [purpose]. Please provide more details about your requirements.",
            "Please provide me with your preferences, and I'll suggest a product that fits your needs."
        ]
    },
    {
        "tag": "order_tracking",
        "patterns": [
            "Where's my order?",
            "Can you track my order?",
            "I haven't received my order yet",
            "Can you provide an update on my order?",
            "How long will it take for my order to arrive?"
        ],
        "responses": [
            "I apologize for the delay. Please provide your order number, and I'll track it for you.",
            "I'd be happy to help you track your order. Please provide your order number.",
            "Sorry to hear that. Please provide your order number, and I'll investigate the issue.",
            "I'd be happy to provide an update on your order. Please provide your order number.",
            "The estimated delivery time for your order is [timeframe]. Please note that this is subject to change."
        ]
    },
    {
        "tag": "return_and_refund",
        "patterns": [
            "I want to return my order",
            "Can I get a refund?",
            "I'm not satisfied with my order",
            "Can I exchange my order?",
            "How do I initiate a return?"
        ],
        "responses": [
            "I apologize that you're not satisfied with your order. Please provide your order number, and I'll guide you through the return process.",
            "Please provide your order number, and I'll assist you with the refund process.",
            "Sorry to hear that. Please provide your order number, and I'll help you with the return or exchange process.",
            "Please provide your order number, and I'll assist you with the exchange process.",
            "To initiate a return, please provide your order number, and I'll guide you through the process."
        ]
    },
    {
        "tag": "payment_inquiry",
        "patterns": [
            "What payment methods do you accept?",
            "Can I pay with [payment method]?",
            "How do I pay for my order?",
            "Can you provide payment instructions?",
            "What's the payment process?"
        ],
        "responses": [
            "We accept various payment methods, including [list payment methods].",
            "Yes, we accept [payment method]. Please follow the payment instructions during checkout.",
            "To pay for your order, please follow the payment instructions during checkout.",
            "Please find the payment instructions below: [payment instructions].",
            "The payment process is as follows: [payment process]."
        ]
    },
    {
        "tag": "shipping_inquiry",
        "patterns": [
            "What's the shipping cost?",
            "Can you provide shipping details?",
            "How long does shipping take?",
            "Do you offer free shipping?",
            "What's the shipping policy?"
        ],
        "responses": [
            "The shipping cost is calculated based on the destination and weight of the package.",
            "Please provide your location, and I'll provide you with shipping details.",
            "Shipping times vary depending on the destination, but we aim to deliver within [timeframe].",
            "Yes, we offer free shipping on orders over [amount].",
            "Our shipping policy is as follows: [shipping policy]."
        ]
    },
    {
        "tag": "product_availability",
        "patterns": [
            "Is this product in stock?",
            "Can you check if this product is available?",
            "How many [products] do you have in stock?",
            "Is this product available in [size/color]?",
            "When will this product be restocked?"
        ],
        "responses": [
            "Yes, this product is in stock and available for immediate shipping.",
            "I've checked, and this product is currently out of stock.",
            "We have [number] of this product in stock.",
            "Yes, this product is available in [size/color].",
            "We expect to restock this product within [timeframe]."
        ]
    },
    {
        "tag": "coupon_and_promotion",
        "patterns": [
            "Do you have any coupons available?",
            "Can I get a discount code?",
            "How do I use a coupon?",
            "What promotions do you have currently?",
            "Can I combine coupons?"
        ],
        "responses": [
            "Yes, we currently have [coupon code] available for [discount].",
            "Please use code [discount code] at checkout to receive [discount].",
            "To use a coupon, simply enter the code at checkout.",
            "We're currently offering [promotion] on [products].",
            "Unfortunately, coupons cannot be combined."
        ]
    },
    {
        "tag": "product_comparison",
        "patterns": [
            "How does this product compare to [product]?",
            "Can you compare [product] and [product]?",
            "What's the difference between [product] and [product]?",
            "Which product is better, [product] or [product]?",
            "How do these products differ?"
        ],
        "responses": [
            "This product has [feature] whereas [product] has [feature].",
            "[Product] and [product] both have [feature], but [product] also has [feature].",
            "The main difference between [product] and [product] is [difference].",
            "Both products have their pros and cons. It ultimately depends on your needs.",
            "These products differ in [difference]."
        ]
    },
    {
        "tag": "product_recommendation_based_on_price",
        "patterns": [
            "What's the best product in this price range?",
            "Can you recommend a product within my budget?",
            "What products do you have for [price]?",
            "I'm looking for a product under [price].",
            "What's the cheapest product you have?"
        ],
        "responses": [
            "Our best product in this price range is [product].",
            "I'd be happy to recommend a product within your budget. Please let me know what you're looking for.",
            "We have [products] available within your budget.",
            "I've found a few options for you: [products].",
            "Our most affordable product is [product]."
        ]
    }
]

# Extract all patterns for suggestions
all_patterns = [pattern for intent in intents for pattern in intent['patterns']]


# Function to preprocess and lemmatize input
def preprocess_input(input_text):
    tokens = word_tokenize(input_text.lower())
    return [lemmatizer.lemmatize(word) for word in tokens]


# Similarity-based intent matching with lemmatization
def find_best_match(input_text):
    best_match = None
    highest_similarity = 0.0
    threshold = 0.6
    processed_input = preprocess_input(input_text)

    for intent in intents:
        for pattern in intent['patterns']:
            processed_pattern = preprocess_input(pattern)
            similarity = SequenceMatcher(None, ' '.join(processed_input), ' '.join(processed_pattern)).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = intent

    if highest_similarity >= threshold:
        return best_match

    return None


# Chatbot logic with expanded matching
def chatbot(input_text):
    # Try direct matching first
    input_words = preprocess_input(input_text)

    for intent in intents:
        for pattern in intent['patterns']:
            pattern_words = preprocess_input(pattern)
            if set(pattern_words).issubset(set(input_words)):
                return random.choice(intent['responses'])

    # Use similarity matching as a fallback
    best_match = find_best_match(input_text)
    if best_match:
        return random.choice(best_match['responses'])

    return "I'm sorry, I didn't understand that. Can you please rephrase?"


# Global counter for unique input keys
counter = 0


# Streamlit app
def main():
    global counter
    st.title("E commerce chatbot ")

    menu = ['Home', 'Conversation History', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.write("Welcome to our E commerce Chatbot. How may I assist you today?")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1

        # Suggest questions to the user
        suggestion = st.selectbox("Suggestions (optional):", ["Type your own"] + all_patterns)

        # Allow free text input
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        # Use the suggestion if no free text is entered
        final_input = user_input or (suggestion if suggestion != "Type your own" else "")

        if final_input:
            response = chatbot(final_input)
            st.text_area('Chatbot:', value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([final_input, response, timestamp])

            # Display a goodbye message if the user says "goodbye"
            if final_input.lower() in ["goodbye", "bye"]:
                st.write("Thank you for chatting with me! Have a wonderful stay!")
                st.stop()

    elif choice == 'Conversation History':
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.write(f"*User: {row[0]}\nChatbot: {row[1]}\n*Time: {row[2]}")
        else:
            st.write("No conversation history found.")

    elif choice == 'About':
        st.subheader("About the E commerce chatbot")
        st.write("""
        Our online shopping chatbot is an innovative, intent-based solution designed to provide customers with a personalized and efficient shopping experience. Built using Python, NLTK, Scikit-Learn, and Streamlit, our chatbot leverages natural language processing (NLP) and machine learning algorithms to understand and respond to customer inquiries.

Intent-Based Classification

Our chatbot uses advanced NLP techniques to classify customer inquiries into specific intents, such as:

- Product information
- Order tracking
- Returns and refunds
- Customer support

By identifying the intent behind each customer inquiry, our chatbot provides accurate and relevant responses, streamlining the shopping experience and improving customer satisfaction.

Key Features
- Intent-based classification
- Natural language processing (NLP)
- Machine learning algorithms
- Personalized responses
- Conversation history
How it Works
1. Customer Input: The customer interacts with the chatbot, providing input in the form of text or voice.
2. Intent Identification: Our chatbot uses NLP and machine learning algorithms to identify the intent behind the customer's input.
3. Response Generation: Based on the identified intent, our chatbot generates a personalized response, providing the customer with relevant information or assistance.
4. Conversation History: Our chatbot stores and displays conversation history, allowing customers to easily track their previous interactions.
Benefits
- Improved customer satisfaction
- Increased efficiency
- Enhanced shopping experience
- Personalized support

By leveraging intent-based classification and NLP, our chatbot provides customers with a more personalized, efficient, and satisfying shopping experience.
    """)
if __name__=="__main__":
    main()