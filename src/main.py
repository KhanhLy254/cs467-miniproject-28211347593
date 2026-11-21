"""Command-line demo for the Pet Adoption Profile and Owner Matcher."""
import argparse
from pathlib import Path
import pandas as pd

from .recommender import PetRecommender
from .evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]

def load_model():
    items = pd.read_csv(ROOT/"data/items.csv")
    ratings = pd.read_csv(ROOT/"data/ratings.csv")
    return PetRecommender(items,ratings)

def print_results(title, frame):
    print("\n" + "="*78)
    print(title)
    print("="*78)
    if frame.empty:
        print("No recommendations available.")
    else:
        print(frame.to_string(index=False))

def run_demo():
    model=load_model()
    query="quiet apartment, first time owner, looking for calm affectionate cat"
    print_results("TEST CASE 1 - ADOPTER QUERY (Top 5)",model.content_query(query,top_n=5))
    print_results("TEST CASE 1B - HYBRID MATCHER (Top 5)",model.hybrid_query(query,user_id="U015",top_n=5))

    selected="P001"
    print_results(f"TEST CASE 2 - SIMILAR PET PROFILES TO {selected}",model.similar_profile(selected,top_n=5))

    # Cold-start pet: zero ratings; append a new profile and rebuild content index.
    new_pet = {
        "item_id":"P041","pet_name":"New Rescue - Maple","species":"Cat","age_group":"Adult",
        "personality_bio":"Maple is a calm affectionate rescued cat who enjoys quiet rooms, gentle attention, and a predictable apartment routine.",
        "lifestyle_tags":"calm, affectionate, quiet, apartment, indoor, first-time-owner"
    }
    model.items=pd.concat([model.items,pd.DataFrame([new_pet])],ignore_index=True)
    from .nlp_pipeline import PetNLP, prepare_documents
    model.nlp=PetNLP().fit(prepare_documents(model.items))
    # Collaborative matrix remains based on existing ratings; new pet has no ratings.
    cold_query="quiet apartment, first time owner, calm affectionate indoor cat"
    cold=model.nlp.similarities(cold_query)
    cold_frame=model.items.copy()
    cold_frame["similarity"]=cold
    cold_frame=cold_frame.sort_values("similarity",ascending=False).head(5)
    print_results("TEST CASE 3 - COLD START (new pet P041 has zero ratings)",cold_frame[["item_id","pet_name","species","similarity"]])
    return model

def main():
    parser=argparse.ArgumentParser(description="Pet adoption NLP and recommendation demo")
    parser.add_argument("--query",help="Custom adopter lifestyle query")
    parser.add_argument("--user",help="Existing adopter ID, e.g. U015")
    parser.add_argument("--evaluate",action="store_true",help="Run offline evaluation and save charts")
    args=parser.parse_args()
    if args.evaluate:
        summary,rmse,mae,n_train,n_test=evaluate()
        print(summary.to_string(index=False))
        print(f"RMSE={rmse:.3f}, MAE={mae:.3f}; train={n_train}, test={n_test}")
        return
    model=load_model()
    if args.query:
        print_results("CUSTOM QUERY - CONTENT BASED",model.content_query(args.query,user_id=args.user,top_n=5))
        print_results("CUSTOM QUERY - HYBRID",model.hybrid_query(args.query,user_id=args.user,top_n=5))
    else:
        run_demo()

if __name__=="__main__":
    main()
