import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
num_images = 200
download_path = "../data/spot_the_camouflaged_coyote"  # Ensure this directory exists
max_attempts = 250  # Max number of attempts to find and download images

# Ensure download path exists
if not os.path.exists(download_path):
    os.makedirs(download_path)

# Setup WebDriver (Assuming Chrome)
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run in background
driver = webdriver.Chrome(options=options)
driver.get("https://www.google.com/search?tbm=isch&q=spot+the+camouflaged+coyote")

images_downloaded = 0
attempt_count = 0

while images_downloaded < num_images and attempt_count < max_attempts:
    img_elements = driver.find_elements("css selector", "img.rg_i")
    for img in img_elements:
        try:
            # Increment the attempt count each time you try to click an image
            attempt_count += 1
            
            clickable_image = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable(img)
            )
            driver.execute_script("arguments[0].scrollIntoView();", clickable_image)
            ActionChains(driver).move_to_element(clickable_image).click(clickable_image).perform()

            large_images = WebDriverWait(driver, 2).until(
                EC.visibility_of_all_elements_located((By.CSS_SELECTOR, 'img.sFlh5c.pT0Scc.iPVvYb'))
            )
            for large_img in large_images:
                src = large_img.get_attribute('src')
                if 'http' in src:  # Filter out base64 encoded images
                    img_data = requests.get(src).content
                    filename = os.path.join(download_path, f'coyote_{images_downloaded}.jpg')
                    with open(filename, 'wb') as handler:
                        handler.write(img_data)
                    print(f"Downloaded {filename}")
                    images_downloaded += 1
                    if images_downloaded >= num_images:
                        break  # Break inner loop
            if images_downloaded >= num_images:
                break  # Break outer loop
        except Exception as e:
            print(f"Encountered an issue: {e}")
            # Optionally, add a short sleep here to reduce the rate of requests
            time.sleep(1)
            continue  # Skip to the next image

    # Optionally, add a short sleep here to reduce the rate of requests
    time.sleep(1)

driver.quit()  # Make sure to close the browser
