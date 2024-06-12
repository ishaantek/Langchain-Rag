# Vector embeddings are text in multi-dimensional vectors
# Words with similar meaning are placed closed to each other in this vector space
# Distance between vectors can be compared with cosine similarity or euclidian distance,
# But there are premade functions already

from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def main():
    # Embeddings for a word
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query("laptop")
    print(f"Vector for 'laptop': {vector}")
    
    # Compare vectors of 2 words
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("laptop", "laptop")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")

if __name__ == "__main__":
    main()