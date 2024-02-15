from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

import config

# Define the table
data = {
    "Cities": ["Paris, France", "London, England", "Lyon, France"],
    "Inhabitants": ["2.161", "8.982", "0.513"],
}
queries = ["What is the population of Lyon?", "What is the population of London?"]


def load_model_and_tokenizer():
    tokenizer = TapasTokenizer.from_pretrained(config.TABULAR_MODEL)
    model = TapasForQuestionAnswering.from_pretrained(config.TABULAR_MODEL)
    return tokenizer, model


def prepare_inputs(data, queries, tokenizer):
    if not type(data) == pd.DataFrame:
        table = pd.DataFrame.from_dict(data)
    else:
        table = data
    inputs = tokenizer(
        table=table, queries=queries, padding="max_length", return_tensors="pt"
    )
    return table, inputs


def generate_predictions(inputs, model, tokenizer):
    """
    Generate predictions for some tokenized input.
    """
    # Generate model results
    outputs = model(**inputs)

    # Convert logit outputs into predictions for table cells and aggregation operators
    (
        predicted_table_cell_coords,
        predicted_aggregation_operators,
    ) = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
    )

    # Return values
    return predicted_table_cell_coords, predicted_aggregation_operators


def postprocess_predictions(
    predicted_aggregation_operators, predicted_table_cell_coords, table
):
    """
    Compute the predicted operation and nicely structure the answers.
    """
    # Process predicted aggregation operators
    aggregation_operators = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    aggregation_predictions_string = [
        aggregation_operators[x] for x in predicted_aggregation_operators
    ]

    # Process predicted table cell coordinates
    answers = []
    for coordinates in predicted_table_cell_coords:
        if len(coordinates) == 1:
            # 1 cell
            answers.append(table.iat[coordinates[0]])
        else:
            # > 1 cell
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
                answers.append(", ".join(cell_values))

    # Return values
    return aggregation_predictions_string, answers


def main(df: pd.DataFrame):
    tokenizer, model = load_model_and_tokenizer()
    table, inputs = prepare_inputs(df, queries, tokenizer)
    predicted_table_cell_coords, predicted_aggregation_operators = generate_predictions(
        inputs, model, tokenizer
    )
    aggregation_predictions_string, answers = postprocess_predictions(
        predicted_aggregation_operators, predicted_table_cell_coords, table
    )
    # TODO: test and refine
