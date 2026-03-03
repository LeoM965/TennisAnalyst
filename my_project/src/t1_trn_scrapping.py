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
from constants import TABLE_CONFIGS, RAW_DATA_DIR, WTA_PLAYERS

def load_players(csv_path=WTA_PLAYERS):
    try:
        dataframe = pd.read_csv(csv_path)
        dataframe['full_name'] = dataframe['first_name'] + ' ' + dataframe['last_name']
        
        return [(name, str(pid)) for name, pid in zip(dataframe['full_name'], dataframe['player_id'])]
    except:
        return []

def create_driver():
    options = webdriver.ChromeOptions()
    
    chrome_args = [
        '--headless',
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--disable-extensions',
        '--disable-logging'
    ]
    
    for argument in chrome_args:
        options.add_argument(argument)
        
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def safe_scrape(driver, player, player_id, table_type, max_retries=3):
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                driver.quit()
                driver = create_driver()
                
            url = f"https://www.tennisabstract.com/cgi-bin/wplayer-more.cgi?p={player_id}/{player.replace(' ', '')}&table={table_type}"
            driver.get(url)
            
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            time.sleep(2)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                
                if len(rows) <= 1:
                    continue
                    
                headers = [cell.get_text(strip=True).replace('\u00a0', ' ') for cell in rows[0].find_all(['th', 'td'])]
                
                if all(key in headers for key in TABLE_CONFIGS[table_type]['key_headers']):
                    data = []
                    
                    for row in rows[1:]:
                        cells = row.find_all(['th', 'td'])
                        if cells:
                            row_data = [player] + [cell.get_text(strip=True).replace('\u00a0', ' ') for cell in cells]
                            data.append(row_data)
                            
                    return driver, headers, data
                    
            return driver, None, None
            
        except:
            if attempt == max_retries - 1:
                return driver, None, None
            time.sleep(3)
            
    return driver, None, None

def scrape_all_data():
    players = load_players()
    if not players:
        return
        
    all_scraped_data = {table_type: [] for table_type in TABLE_CONFIGS}
    headers_initialized = {table_type: False for table_type in TABLE_CONFIGS}
    
    driver = create_driver()
    
    try:
        for index, (player, player_id) in enumerate(players):
            for table_type in TABLE_CONFIGS:
                driver, headers, data = safe_scrape(driver, player, player_id, table_type)
                
                if headers and data:
                    if not headers_initialized[table_type]:
                        all_scraped_data[table_type].append(['Player'] + headers)
                        headers_initialized[table_type] = True
                        
                    for row in data:
                        if len(row) > 1 and row[1] != 'Match':
                            all_scraped_data[table_type].append(row)
                            
            print(f"Processed: {index + 1}/{len(players)} - {player}")
            time.sleep(1)
            
    finally:
        try:
            driver.quit()
        except:
            pass
            
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        
    for table_type, records in all_scraped_data.items():
        if records and len(records) > 1:
            output_df = pd.DataFrame(records[1:], columns=records[0])
            output_file = os.path.join(RAW_DATA_DIR, TABLE_CONFIGS[table_type]['filename'])
            output_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    scrape_all_data()
