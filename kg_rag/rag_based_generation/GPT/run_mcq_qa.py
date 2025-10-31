"""
This script takes the MCQ style questions from the csv file and save the result as another csv file.
Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
"""

from kg_rag.utility import *
import sys
import argparse
import logging

logger = logging.Logger("idk_name")

# python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-2.0-flash --mode 1
# python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-2.0-flash --mode 2 --prior_knowledge_path prior_knowledge.txt
# python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-2.0-flash --mode 3 --prior_knowledge_path prior_knowledge.txt


from tqdm import tqdm

QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(
    config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"]
)
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(
    config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"]
)
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data[
    "SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"
]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data[
    "SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"
]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]


vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(
    SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL
)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False


parser = argparse.ArgumentParser()
parser.add_argument("chat_model_id", type=str, help="Chat model ID to use")
parser.add_argument(
    "--mode",
    type=str,
    help="""### MODE 0: Original KG_RAG                     
### MODE 1: jsonlize the context from KG search ### 
### MODE 2: Add the prior domain knowledge      ### 
### MODE 3: Combine MODE 1 & 2                  ###
### MODE 4: Same as MODE 0, but filter context to only include those with genes mentioned in the question

""",
)
parser.add_argument(
    "--start_index",
    type=int,
    default=0,
    help="Start index of questions (0-indexed)",
)
parser.add_argument(
    "--prior_knowledge",
    type=str,
    help="Prior knowledge you want to inject into the model",
)
parser.add_argument(
    "--prior_knowledge_path", type=str, help="Path to the prior knowledge file"
)
parser.add_argument(
    "--end_index",
    type=int,
    default=306,
    help="End index of questions (0-indexed, exclusive)",
)


args = parser.parse_args()
CHAT_MODEL_ID = args.chat_model_id
CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID
timestamp = time.time_ns()
prior_knowledge = args.prior_knowledge
prior_knowledge_path = args.prior_knowledge_path
start_index = args.start_index
end_index = args.end_index
if prior_knowledge_path:
    with open(prior_knowledge_path, "r") as file:
        prior_knowledge = file.read()

MODE = args.mode
save_name = (
    "_".join(CHAT_MODEL_ID.split("-")) + f"_kg_rag_based_mcq_{MODE}_{timestamp}.csv"
)

VALID_MODES = ["0", "1", "2", "3", "4"]
if prior_knowledge is None and (MODE == "2" or MODE == "3"):
    raise ValueError("Need prior knowledge when in modes 2 or 3")


assert MODE in VALID_MODES, "Invalid mode. Please choose from 0, 1, 2, or 3."


def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []

    for index, row in tqdm(question_df.iterrows(), total=306):
        try:
            if index < start_index - 1:
                answer_list.append((row["text"], row["correct_node"], "Skipped"))
                continue
            if index > end_index:
                break
            # try:
            question = row["text"]

            ### MODE 0: Original KG_RAG                     ###
            if MODE == "0" or MODE == "2" or MODE == "4":
                context = retrieve_context(
                    row["text"],
                    vectorstore,
                    embedding_function_for_context_retrieval,
                    node_context_df,
                    CONTEXT_VOLUME,
                    QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
                    edge_evidence,
                    model_id=CHAT_MODEL_ID,
                    mode_4=MODE == "4",
                )

            elif MODE == "1" or MODE == "3":
                context = retrieve_context_json(
                    row["text"],
                    vectorstore,
                    embedding_function_for_context_retrieval,
                    node_context_df,
                    CONTEXT_VOLUME,
                    QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
                    edge_evidence,
                    model_id=CHAT_MODEL_ID,
                )

            else:

                raise ValueError("Invalid mode. Please choose from 0, 1, 2, 3, or 4.")

            if MODE == "0" or MODE == "1" or MODE == "4":
                prompt = "Context: " + context + "\n" + "Question: " + question

            elif MODE == "2" or MODE == "3":
                ### MODE 2: Add the prior domain knowledge      ###
                ### Please implement the second strategy here   ###
                print("context", context)
                print("prior_knowledge", prior_knowledge)
                print("question", question)
                prompt = (
                    "Context: "
                    + context
                    + "\n"
                    + "IMPORTANT NOTES:\n"
                    + prior_knowledge
                    + "\n"
                    + "Question: "
                    + question
                )
                logger.info(f"Prompt for mode {MODE}: {prompt}")

            else:
                raise ValueError("MODE invalid")

            output = get_Gemini_response(prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            answer_list.append((row["text"], row["correct_node"], output))
            # except Exception as e:
            #     print("Error in processing question: ", row["text"])
            #     print("Error: ", e)

        except Exception as e:
            print("Error in processing question: ", row["text"])
            print("Error: ", e)
            time.sleep(10)

        answer_df = pd.DataFrame(
            answer_list, columns=["question", "correct_answer", "llm_answer"]
        )
        output_file = os.path.join(SAVE_PATH, save_name)
        answer_df.to_csv(output_file, index=False, header=True)
        print("Save the model outputs in ", output_file)
        print("Completed in {} min".format((time.time() - start_time) / 60))


if __name__ == "__main__":
    main()
