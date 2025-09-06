import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time


def load_players(csv_path='wta_players.csv'):
    try:
        df = pd.read_csv(csv_path)
        df['full_name'] = df['first_name'] + ' ' + df['last_name']
        return [(name, str(pid)) for name, pid in zip(df['full_name'], df['player_id'])]
    except:
        return []


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def scrape_player(player, player_id, driver, retries=2):
    for attempt in range(retries):
        try:
            url = f"https://www.tennisabstract.com/cgi-bin/wplayer-more.cgi?p={player_id}/{player.replace(' ', '')}&table=winners-errors"
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                if len(rows) > 1:
                    headers = [cell.get_text(strip=True) for cell in rows[0].find_all(['th', 'td'])]
                    if 'Winners' in headers and 'UFEs' in headers and 'Match' in headers:
                        data = []
                        for row in rows[1:]:
                            cells = row.find_all(['th', 'td'])
                            if cells:
                                row_data = [player] + [cell.get_text(strip=True) for cell in cells]
                                data.append(row_data)
                        return headers, data
            return [], []
        except:
            if attempt == retries - 1:
                return [], []
            time.sleep(1)
    return [], []


def main():
    players = load_players()
    if not players:
        return

    driver = setup_driver()
    all_data = []
    headers_set = False
    processed = 0

    try:
        for player, player_id in players:
            headers, data = scrape_player(player, player_id, driver)
            if data:
                if not headers_set:
                    all_data.append(['Player'] + headers)
                    headers_set = True
                all_data.extend([row for row in data if len(row) > 1 and row[1] != 'Match'])

            processed += 1
            print(f"Procesat: {processed}/{len(players)} - {player}")
            time.sleep(0.5)
    finally:
        driver.quit()

    if all_data and len(all_data) > 1:
        df = pd.DataFrame(all_data[1:], columns=all_data[0])
        df.to_csv('wta_winners_unforced_errors.csv', index=False)
        print(f"Date salvate in wta_winners_unforced_errors.csv - {len(df)} randuri")
    else:
        print("Nu s-au gasit date")


if __name__ == "__main__":
    main()