import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os

def load_players(csv_path='wta_players.csv'):
    try:
        dataframe = pd.read_csv(csv_path)
        dataframe['full_name'] = dataframe['first_name'] + ' ' + dataframe['last_name']
        
        return [(name, str(pid)) for name, pid in zip(dataframe['full_name'], dataframe['player_id'])]
    except:
        return []

def configure_driver():
    chrome_options = webdriver.ChromeOptions()
    
    chrome_args = [
        '--headless',
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--disable-extensions',
        '--disable-logging'
    ]
    
    for argument in chrome_args:
        chrome_options.add_argument(argument)
        
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    chrome_service = Service(ChromeDriverManager().install())
    
    return webdriver.Chrome(service=chrome_service, options=chrome_options)

def scrape_winners_errors(player_name, player_id, driver, max_retries=3):
    for attempt in range(max_retries):
        try:
            url = f"https://www.tennisabstract.com/cgi-bin/wplayer-more.cgi?p={player_id}/{player_name.replace(' ', '')}&table=winners-errors"
            driver.get(url)
            
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            time.sleep(1)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                
                if len(rows) > 1:
                    headers = [cell.get_text(strip=True) for cell in rows[0].find_all(['th', 'td'])]
                    
                    if all(key in headers for key in ['Winners', 'UFEs', 'Match']):
                        scraped_data = []
                        
                        for row in rows[1:]:
                            cells = row.find_all(['th', 'td'])
                            if cells:
                                row_data = [player_name] + [cell.get_text(strip=True) for cell in cells]
                                scraped_data.append(row_data)
                                
                        return headers, scraped_data
            return None, None
            
        except:
            if attempt == max_retries - 1:
                return None, None
            time.sleep(2)
            
    return None, None

def main():
    players_to_scrape = load_players()
    if not players_to_scrape:
        return
        
    chrome_driver = configure_driver()
    aggregated_results = []
    has_headers = False
    
    try:
        for index, (player, player_id) in enumerate(players_to_scrape):
            headers, player_data = scrape_winners_errors(player, player_id, chrome_driver)
            
            if player_data:
                if not has_headers:
                    aggregated_results.append(['Player'] + headers)
                    has_headers = True
                    
                for row in player_data:
                    if len(row) > 1 and row[1] != 'Match':
                        aggregated_results.append(row)
                        
            print(f"Processed: {index + 1}/{len(players_to_scrape)} - {player}")
            time.sleep(0.5)
            
    finally:
        chrome_driver.quit()
        
    if len(aggregated_results) > 1:
        output_df = pd.DataFrame(aggregated_results[1:], columns=aggregated_results[0])
        output_df.to_csv('wta_winners_unforced_errors.csv', index=False)
        print(f"Scraping complete. Saved {len(output_df)} rows.")

if __name__ == "__main__":
    main()