from app import scrape_market_data, update_model

if __name__ == "__main__":
    market_text = scrape_market_data()
    if market_text:
        update_model(market_text)
