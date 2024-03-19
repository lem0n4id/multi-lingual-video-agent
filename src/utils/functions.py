from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_video_script(url: str) -> list:
    """
    Get the script of a YouTube video by its URL and return it as a list of strings
    """
    
    if not url:
        raise ValueError("YouTube video url is required")
    
    video_url = urlparse("https://www.youtube.com/watch?v=1c9iyoVIwDs")
    video_query = parse_qs(video_url.query)
    
    video_id = None
    
    if "v" in video_query:
        video_id = video_query["v"][0]
    else:
        raise ValueError("Invalid YouTube video url")
    
    if not video_id:
        raise ValueError("Invalid YouTube video url")
    
    video_script = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    
    video_text = ""

    for segment in video_script:
        video_text += segment['text'] + " "
        
    return [segment['text'] for segment in video_script if segment["text"]]

def chunk_text(text: list, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
    """
    Split a list of strings into chunks of strings
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    texts = text_splitter.create_documents(text)
    
    return texts
