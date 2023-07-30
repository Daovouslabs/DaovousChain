import re

import chainlit as cl

async def send_medias_message(message):
    image_urls, audio_urls, video_urls = extract_medias(message)
    for image_url in image_urls:
        await cl.Image(url=image_url, name=image_url, display="inline").send()
    
    for audio_url in audio_urls:
        await cl.Audio(url=audio_url, name=audio_url, display="inline").send()

    for video_url in video_urls:
        await cl.Video(url=video_url, name=video_url, display="inline").send()
        pass

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

def is_media(url):
    pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(jpg|jpeg|tiff|gif|png|flac|wav|mp4)")
    res = pattern.search(url)
    return False if res is None else True