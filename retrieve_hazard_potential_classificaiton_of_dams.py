from selenium import webdriver
from selenium.webdriver.common.by import By
#from selenium.webdriver.chrome.service import Service
#from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(executable_path="D:\Drivers\chromedriver.exe", options=chrome_options)
#driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
#driver = webdriver.Chrome(executable_path="D:\Drivers\chromedriver.exe")

df = pd.read_csv(
    "https://raw.githubusercontent.com/I-GUIDE/population_vulnerable_to_dam_failure/main/nid_available_scenario.csv")
df = df[(df['MH_F'] == True) & (df['TAS_F'] == True) & (df['NH_F'] == True)]

risk_list = []
date_list = []
for index, id in enumerate(df['ID']):
    try:
        search_url = f"https://nid.usace.army.mil/#/dams/system/{id}/risk"
        driver.get(search_url)
        driver.implicitly_wait(5)

        risk_path = driver.find_element(
            By.XPATH, "//input[@ng-show='vm.isReadonly'][@class='nld-input'][@ng-attr-title='{{vm.optionsDisplay(vm.model)}}']")
        date_path = driver.find_element(
            By.XPATH, "//input[@type='date'][@ng-model='vm.model'][@ng-readonly='vm.isReadonly']")
        risk = risk_path.get_attribute("value")
        date = date_path.get_attribute("value").replace("-", "")
        print(f"{id}  {risk}  {date}")
    except Exception as e:
        print(e)
        risk_list.append('None')
        date_list.append('None')
        continue
    risk_list.append(risk)
    date_list.append(date)
    time.sleep(1)
driver.quit()

df['Risk'] = risk_list
df['Date_Assessed'] = date_list
df.to_csv(r'C:\Users\jrh25\Desktop\dam_data.csv', index=False)