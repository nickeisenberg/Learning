from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()

sign_in = 'https://login.fidelity.com/ftgw/Fas/Fidelity/RtlCust/Login/Init?AuthRedUrl=https%3A//researchtools.fidelity.com/ftgw/mloptions/goto/optionChain'

driver.find_element(By.ID, "userId-input").send_keys('')
driver.find_element(By.ID, "password").send_keys('')
driver.find_element(By.ID, "fs-login-button").click()

url = 'https://researchtools.fidelity.com/ftgw/mloptions/goto/optionChain'
driver.get(url)

driver.find_element(By.ID, "symbol").send_keys('SPY')

