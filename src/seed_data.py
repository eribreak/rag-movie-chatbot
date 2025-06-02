import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4
from crawl import crawl_web
from langchain_ollama import OllamaEmbeddings

load_dotenv()

def load_data_from_local(filename: str, directory: str) -> tuple:

    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(f'Data loaded from {file_path}')
    # Chuyển tên file thành tên tài liệu (bỏ đuôi .json và thay '_' bằng khoảng trắng)
    return data, filename.rsplit('.', 1)[0].replace('_', ' ')

def convert_movie_json(data, doc_name):

    # Nếu là dict đơn lẻ thì chuyển thành list
    if isinstance(data, dict):
        data = [data]
    result = []
    for item in data:
        # Ghép các trường thành page_content
        genres = ', '.join([g['name'] for g in item.get('genres', [])])
        keywords = ', '.join([k['name'] for k in item.get('keywords', [])])
        companies = ', '.join([c['name'] for c in item.get('production_companies', [])])
        countries = ', '.join([c['name'] for c in item.get('production_countries', [])])
        spoken_languages = ', '.join([l['name'] for l in item.get('spoken_languages', [])])
        page_content = f"Title: {item.get('title', '')}\n" \
            f"Original Title: {item.get('original_title', '')}\n" \
            f"Tagline: {item.get('tagline', '')}\n" \
            f"Overview: {item.get('overview', '')}\n" \
            f"Genres: {genres}\n" \
            f"Keywords: {keywords}\n" \
            f"Production Companies: {companies}\n" \
            f"Production Countries: {countries}\n" \
            f"Spoken Languages: {spoken_languages}\n" \
            f"Release Date: {item.get('release_date', '')}\n" \
            f"Status: {item.get('status', '')}\n" \
            f"Homepage: {item.get('homepage', '')}\n" \
            f"Budget: {item.get('budget', '')}\n" \
            f"Revenue: {item.get('revenue', '')}\n" \
            f"Runtime: {item.get('runtime', '')}\n" \
            f"Vote Average: {item.get('vote_average', '')}\n" \
            f"Vote Count: {item.get('vote_count', '')}\n"
        metadata = {
            'source': item.get('homepage', ''),
            'content_type': 'movie',
            'title': item.get('title', ''),
            'description': item.get('overview', ''),
            'language': item.get('original_language', 'en'),
            'doc_name': doc_name,
            'release_date': item.get('release_date', ''),
            'genres': genres,
            'keywords': keywords,
            'production_companies': companies,
            'production_countries': countries,
            'spoken_languages': spoken_languages,
            'budget': item.get('budget', 0),
            'revenue': item.get('revenue', 0),
            'runtime': item.get('runtime', 0),
            'vote_average': item.get('vote_average', 0),
            'vote_count': item.get('vote_count', 0),
        }
        result.append({
            'page_content': page_content,
            'metadata': metadata
        })
    return result

def ensure_metadata_fields(metadata, required_fields):
    """
    Đảm bảo metadata là dict và có đủ các trường cần thiết, nếu thiếu thì gán giá trị mặc định là ''.
    """
    if metadata is None:
        metadata = {}
    for field in required_fields:
        if field not in metadata or metadata[field] is None:
            metadata[field] = ''
    return metadata

def seed_milvus(URI_link: str, collection_name: str, filename: str, directory: str, use_ollama: bool = False, data_type: str = "default") -> Milvus:

    # Khởi tạo model embeddings tùy theo lựa chọn
    if use_ollama:
        embeddings = OllamaEmbeddings(
            model="llama2"  
        )
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Đọc dữ liệu từ file local
    local_data, doc_name = load_data_from_local(filename, directory)

    # Chuyển đổi dữ liệu thành danh sách các Document với giá trị mặc định cho các trường
    if data_type == "movie":
        processed_data = convert_movie_json(local_data, doc_name)
    else:
        processed_data = local_data

    required_metadata_fields = [
        'source', 'content_type', 'title', 'description', 'language', 'doc_name',
        'release_date', 'genres', 'keywords', 'production_companies',
        'production_countries', 'spoken_languages', 'budget', 'revenue',
        'runtime', 'vote_average', 'vote_count'
    ]

    documents = [
        Document(
            page_content=doc.get('page_content') or '',
            metadata=ensure_metadata_fields(doc.get('metadata') or {}, required_metadata_fields)
        )
        for doc in processed_data
    ]

    print('documents: ', documents)

    # Tạo ID duy nhất cho mỗi document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Khởi tạo và cấu hình Milvus
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        drop_old=True  # Xóa data đã tồn tại trong collection
    )
    # Thêm documents vào Milvus
    vectorstore.add_documents(documents=documents, ids=uuids)
    print('vector: ', vectorstore)
    return vectorstore

def seed_milvus_live(URL: str, URI_link: str, collection_name: str, doc_name: str, use_ollama: bool = False) -> Milvus:
    """
    Hàm crawl dữ liệu trực tiếp từ URL và tạo vector embeddings trong Milvus
    Args:
        URL (str): URL của trang web cần crawl dữ liệu
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection trong Milvus
        doc_name (str): Tên định danh cho tài liệu được crawl
        use_ollama (bool): Sử dụng Ollama embeddings thay vì OpenAI
    """
    if use_ollama:
        embeddings = OllamaEmbeddings(
            model="llama2"  # hoặc model khác mà bạn đã cài đặt
        )
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    documents = crawl_web(URL)

    # Cập nhật metadata cho mỗi document với giá trị mặc định
    for doc in documents:
        metadata = {
            'source': doc.metadata.get('source') or '',
            'content_type': doc.metadata.get('content_type') or 'text/plain',
            'title': doc.metadata.get('title') or '',
            'description': doc.metadata.get('description') or '',
            'language': doc.metadata.get('language') or 'en',
            'doc_name': doc_name,
            # 'start_index': doc.metadata.get('start_index') or 0
        }
        doc.metadata = metadata

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        drop_old=True
    )
    vectorstore.add_documents(documents=documents, ids=uuids)
    print('vector: ', vectorstore)
    return vectorstore

def connect_to_milvus(URI_link: str, collection_name: str) -> Milvus:

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vectorstore

def main():

    # Test seed_milvus với dữ liệu local
    seed_milvus('http://localhost:19530', 'data_test', 'stack.json', 'data', use_ollama=False)
    # Test seed_milvus_live với URL trực tiếp
    # seed_milvus_live('https://www.stack-ai.com/docs', 'http://localhost:19530', 'data_test_live', 'stack-ai', use_ollama=False)

# Chạy main() nếu file được thực thi trực tiếp
if __name__ == "__main__":
    main()
