import re

import mimetypes

def extract_medias(message):
    image_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(jpg|jpeg|tiff|gif|png)")
    image_urls = []
    for match in image_pattern.finditer(message):
        if match.group(0) not in image_urls:
            image_urls.append(match.group(0))

    audio_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(flac|wav)")
    audio_urls = []
    for match in audio_pattern.finditer(message):
        if match.group(0) not in audio_urls:
            audio_urls.append(match.group(0))

    video_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(mp4)")
    video_urls = []
    for match in video_pattern.finditer(message):
        if match.group(0) not in video_urls:
            video_urls.append(match.group(0))

    return image_urls, audio_urls, video_urls

def get_url_type(url):
    mimetype, encoding = mimetypes.guess_type(url)
    return mimetype

def is_media(url):
    pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(jpg|jpeg|tiff|gif|png|flac|wav|mp4)")
    res = pattern.search(url)
    return False if res is None else True