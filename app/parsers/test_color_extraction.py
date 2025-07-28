from color_extractor import get_color_label_from_image_url

if __name__ == "__main__":
    image_url = "https://pcdn.goldapple.ru/p/p/19000389206/web/746f6e658ddc5ca6fd446e9mobile.jpg"
    result = get_color_label_from_image_url(image_url)
    print(result)
