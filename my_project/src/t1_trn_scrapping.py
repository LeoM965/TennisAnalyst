import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

TABLE_CONFIGS = {
   'mcp-rally': {'key_headers': ['Match', 'RLen-Serve', 'RLen-Return'], 'filename': 'wta_mcp_rally.csv'},
   'mcp-serve': {'key_headers': ['Match', 'Unret%'], 'filename': 'wta_mcp_serve.csv'},
   'mcp-tactics': {'key_headers': ['Match', 'SnV Freq', 'SnV W%'], 'filename': 'wta_mcp_tactics.csv'},
   'mcp-return': {'key_headers': ['Match', 'RiP%'], 'filename': 'wta_mcp_return.csv'}
}

def load_players(csv_path='wta_players.csv'):
   try:
       df = pd.read_csv(csv_path)
       df['full_name'] = df['first_name'] + ' ' + df['last_name']
       return [(name, str(pid)) for name, pid in zip(df['full_name'], df['player_id'])]
   except:
       return []

def create_driver():
   options = webdriver.ChromeOptions()
   options.add_argument('--headless')
   options.add_argument('--no-sandbox')
   options.add_argument('--disable-dev-shm-usage')
   options.add_argument('--disable-gpu')
   options.add_argument('--disable-extensions')
   options.add_argument('--disable-logging')
   options.add_argument('--disable-background-timer-throttling')
   options.add_argument('--disable-backgrounding-occluded-windows')
   options.add_argument('--disable-renderer-backgrounding')
   options.add_argument('--disable-features=TranslateUI')
   options.add_argument('--disable-ipc-flooding-protection')
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
               try:
                   driver.quit()
               except:
                   pass
               driver = create_driver()
               return driver, None, None
           time.sleep(3)

   return driver, None, None

def scrape_all_data():
   players = load_players()
   if not players:
       return

   all_data = {table_type: [] for table_type in TABLE_CONFIGS}
   headers_saved = {table_type: False for table_type in TABLE_CONFIGS}

   driver = create_driver()

   for i, (player, player_id) in enumerate(players):
       print(f"Procesare {i+1}/{len(players)}: {player}")

       for table_type in TABLE_CONFIGS:
           try:
               driver, headers, data = safe_scrape(driver, player, player_id, table_type)

               if headers and data:
                   if not headers_saved[table_type]:
                       all_data[table_type].append(['Player'] + headers)
                       headers_saved[table_type] = True

                   for row in data:
                       if len(row) > 1 and row[1] != 'Match':
                           all_data[table_type].append(row)

               time.sleep(1)

           except:
               continue

   try:
       driver.quit()
   except:
       pass

   for table_type, data in all_data.items():
       if data and len(data) > 1:
           df = pd.DataFrame(data[1:], columns=data[0])
           filename = TABLE_CONFIGS[table_type]['filename']
           df.to_csv(filename, index=False)
           print(f"Salvat: {filename} ({len(df)} randuri)")

if __name__ == "__main__":
   scrape_all_data()