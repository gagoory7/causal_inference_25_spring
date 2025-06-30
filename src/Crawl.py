import os
import time
import datetime
from datetime import datetime, timedelta

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def driver_setting():
    service = Service(ChromeDriverManager().install(), log_output=subprocess.DEVNULL)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=chrome")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")  
    options.add_experimental_option("excludeSwitches", ["enable-logging"])  
    options.set_capability("goog:loggingPrefs", {"browser": "OFF"})
    
    driver = webdriver.Chrome(service=service, options=options)

    return driver

def try_multiple_xpaths(driver, xpaths, url, attr="text"):
    for xpath in xpaths:
        try:
            element = driver.find_element(By.XPATH, xpath)
            if attr == "text":
                return element.text.strip()
            else:
                return element.get_attribute(attr)
        except Exception:
            continue
    return None


def information_crawl(url) :
    infor_driver = driver_setting()
    infor_driver.get(url)
    time.sleep(0.5) 

    WebDriverWait(infor_driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "body")))


    if any(x in infor_driver.current_url for x in ["sports", "entertain"]):
        return None
    
    news_data = {"url": url}

    section_xpath = [
    "/html/body/div/header/div/div[2]/div/div/ul/li[contains(@class, 'is_active')]/a/span[@class='Nitem_link_menu']",
    "//*[@id='_LNB']/ul/li[contains(@class, 'is_active')]"
    ]
    
    news_data["section"] = try_multiple_xpaths(infor_driver,section_xpath,url)


    title_xpaths = [
        '//*[@id="title_area"]/span'
    ]
        
    news_data["title"] = try_multiple_xpaths(infor_driver, title_xpaths,url)

    reporter_xpaths = [
        '//*[@id="ct"]/div[1]/div[3]/div[2]/a/em',
        '//*[@id="ct"]/div[2]/div[3]/div[2]/a/em',
    ]

    news_data["reporter"] = try_multiple_xpaths(infor_driver, reporter_xpaths,url)
    
    reporter_link_xpaths = [
        '//*[@id="_JOURNALIST_LAYER"]/ul/li/div/a',
        '//*[@id="ct"]/div[1]/div[3]/div[2]/a',
        '//*[@id="ct"]/div[2]/div[3]/div[2]/a'
    ]

    news_data["reporter_link"] = try_multiple_xpaths(infor_driver, reporter_link_xpaths, url,"href")

    date_xpaths = [
        '//*[@id="ct"]/div[1]/div[3]/div[1]/div[1]/span',
        '//*[@id="ct"]/div[2]/div[3]/div[1]/div[1]/span'
    ]

    news_data["date"] = try_multiple_xpaths(infor_driver, date_xpaths,url)

    reaction_xpaths = [
        '//*[@id="commentFontGroup"]/div[1]/div/a/span[2]',
        '//*[@id="commentFontGroup"]/div[1]/div/a'
    ]

    news_data["reactions"] = try_multiple_xpaths(infor_driver, reaction_xpaths,url)
    
    comment_xpaths = [
        '//*[@id="comment_count"]'
    ]

    news_data["comments"] = try_multiple_xpaths(infor_driver, comment_xpaths,url)
    
    content_xpaths = [
        '//*[@id="dic_area"]'
    ]

    news_data["content"] = try_multiple_xpaths(infor_driver, content_xpaths,url)

    infor_driver.quit()
    
    time.sleep(0.5)

    return news_data


def crawl_by_date(k, v, date_str):
    driver = driver_setting()
    data_list = []
    erro_list = []
    prev_page = 0
    page = 1

    file = f"data/crawl/{k}_{date_str}"

    if os.path.exists(file+'.csv'):
        return
    
    while True:
        url = f"https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&oid={v}&date={date_str}&page={page}"
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        x_path = "/html/body/div[1]/table/tbody/tr/td[2]/div/div[3]/strong"
        current_page = driver.find_element(By.XPATH, x_path).text.strip()

        if prev_page == current_page:
            break
        else:
            prev_page = current_page

        ### debugging
        # print(f'{k} - date: {date_str}, page: {page}')

        for i in range(1, 3):
            check_non_link = 0
            for j in range(1, 11):
                xpath = f"/html/body/div[1]/table/tbody/tr/td[2]/div/div[2]/ul[{i}]/li[{j}]/dl/dt[2]/a"
                try:
                    element = driver.find_element(By.XPATH, xpath)
                    link = element.get_attribute("href")
                except:
                    check_non_link += 1
                    if check_non_link == 20:
                        print(url)
                        print(f'{k},{v} 이전페이지와 같은데 잡지를 못함 파악해야함')
                        break
                try:
                    info = information_crawl(link)
                    if info:
                        info['media'] = k
                        data_list.append(info)
                except Exception:
                    print(f"[ERROR] {link}")
                    erro_list.append(link)
                
        break

    driver.quit()
    pd.DataFrame(data_list).to_csv(f'{file}.csv', index=False, encoding='utf-8-sig')

    if erro_list : 
        pd.DataFrame(erro_list).to_csv(f'{file}_error.csv',index=False,encoding='utf-8-sig')


def start_crwal(media_company,date,days,workers): 

    start_date = datetime.strptime(str(date), "%Y%m%d")

    dates = [(start_date - timedelta(days=i)).strftime("%Y%m%d") for i in range(days)]

    tasks = []
    for k, v in media_company.items():
        for date_str in dates:
            tasks.append((k, v, date_str))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(crawl_by_date, k, v, date_str) for k, v, date_str in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Crawling"):
            pass

