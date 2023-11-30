import pickle
from sentence_transformers import SentenceTransformer
from flask import Flask,request,jsonify
from flask_cors import CORS
from flask_cors import cross_origin
from qdrant_client import QdrantClient
import dill
BERT_MODEL_PATH = "reader.pkl"
ENC_PATH  = "embedding_model.pkl"
QDRANT_PATH  = "collection.pkl"


with open(BERT_MODEL_PATH, 'rb') as file:
    bert = pickle.load(file)
with open("collection.pkl", "rb") as file:
    qdrant_client = pickle.load(file)



 
with open(ENC_PATH, 'rb') as file:
    st_encoder = pickle.load(file)

collection_name = "qdrant_db"
def get_relevant_context(question: str, top_k: int) -> list[str]:
    """
    will return contexts contexts close to query

    Args:
        question (str): What do we want to know?
        top_k (int): top k results will be added

    Returns:
        context (List[str]):
    """
    try:
        encoded_query = st_encoder.encode(question).tolist()
        result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=encoded_query,
            limit=top_k,
        )  # search qdrant collection for context passage with the answer

        context = [
            [context.payload["metadata"], context.payload["page_content"]] for context in result
        ]
        return context
    except Exception as e:
        print("Here is the error")

def extract_answer(question: str, context: list[str]):
    """
    Extract the answer from the context for a given question

    Args:
        question (str): _description_
        context (list[str]): _description_
    """
    results = []
    for c in context:
        answer = bert(question=question, context=c[1] )
        answer["product"] = c[0]
        results.append(answer)
        print()

    sorted_result = sorted(results, key=lambda x: x["score"], reverse=True)
    for i in range(len(sorted_result)):
        _out = sorted_result[i]["answer"]
        _prod = sorted_result[i]["product"]
        _sco = sorted_result[i]["score"]
#         print(f"{i+1}", end=" ")
        print(f"QUERY INPUT: {question}")
        print(f"OUTPUT: {_out} \nPREDICTION SCORE {_sco}\n\nReferred Product: {_prod}\n\n")
        return question,_out,_sco,_prod



app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5000"}})


# by default get request
@app.route('/')
def home():
    return {
        "api_name": "BigBasket-s-products-Query-Engine"
    }

@app.route('/test', methods=['POST'])
@cross_origin(origin='http://localhost:5000', headers=['Content-Type', 'Authorization'])
def test():
    data = request.get_json()

    _ques  = data["question"]
    c = get_relevant_context(_ques, top_k=2)
    _ques, _out, _sco, _prod = extract_answer(_ques, c)

    result = {
        "_ques": _ques,
        "_out":_out,
        "_sco":_sco,
        "_prod":_prod
    }

    return {
        "type":"output",
        "status":200,
        "payload": result,
     }

if __name__ == '__main__':
    app.run(debug=True)