import argparse
import pandas as pd
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score

nltk.download('wordnet')

def compute_metrics(input_csv):
    df = pd.read_csv(input_csv)
    
    references = df["target"].tolist()
    hypotheses = df["predictions"].tolist()

    # Compute TER using SacreBLEU's corpus_score
    ter_metric = sacrebleu.metrics.TER()
    ter_score_obj = ter_metric.corpus_score(hypotheses, [references])
    ter_score = ter_score_obj.score  # extract the numerical score
    
    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()  # tokenize reference
        hyp_tokens = hyp.split()  # tokenize hypothesis
        score = meteor_score([ref_tokens], hyp_tokens)
        meteor_scores.append(score)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    
    print("TER score: {:.2f}".format(ter_score))
    print("Average METEOR score: {:.2f}".format(avg_meteor))

def main():
    parser = argparse.ArgumentParser(description="Compute TER and METEOR scores from a CSV file.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="de_nl_dev_predictions.csv",
        help="Path to input CSV file containing 'target' and 'predictions' columns."
    )
    args = parser.parse_args()
    compute_metrics(args.input_csv)

if __name__ == "__main__":
    main()

#de-nl-pred = TER 78.82; METEOR: 0.29; 
