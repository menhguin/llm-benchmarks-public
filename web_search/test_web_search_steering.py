import anthropic
from openai import OpenAI
from dotenv import load_dotenv
import csv
import json
import os
import sys
import asyncio
import glob
from datetime import datetime
from typing import List, Dict, Any, Tuple, NamedTuple
from dataclasses import dataclass

# Load environment variables from .env file
load_dotenv()

# Global variable for print responses
print_responses = True

# Constants
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_SYSTEM_PROMPT = "Search the internet for up-to-date information."
DEFAULT_API = "OpenAI"

@dataclass
class TestCase:
    web_search_expected: bool
    score: float
    test_question: str
    context: str

class DetailedTestResult(NamedTuple):
    model: str
    api_source: str
    precision: float
    recall: float
    success_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_cases: int
    comments: str
    score_level_stats: Dict[int, Dict[str, Any]]

# Initialize API clients with better error handling
try:
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
except KeyError:
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please make sure you have set the OPENAI_API_KEY in your .env file or environment.")
    sys.exit(1)

try:
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
except KeyError:
    print("Warning: ANTHROPIC_API_KEY not found in environment variables.")
    print("Anthropic models will not be available for testing.")
    anthropic_client = None
except Exception as e:
    print(f"Error initializing Anthropic client: {str(e)}")
    anthropic_client = None

# Web search tools (moved to a separate configuration in a real-world scenario)
web_search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet for up-to-date information using the Vespa search enginer.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        },
    },
}

web_search_tool_anthropic = {
    "name": "web_search",
    "description": "Search the internet for up-to-date information using the Vespa search engine.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            }
        },
        "required": ["query"]
    }
}

def load_available_models() -> Dict[str, List[Dict[str, str]]]:
    available_models = {}
    try:
        with open("web_search/test_system_prompts.csv", "r", newline='') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                test_model = row.get("test_model", "").strip()
                test_system_prompt = row.get("test_system_prompt", "").strip()
                api_source = row.get("api_source", "").strip()
                comments = row.get("comments", "").strip()
                if test_model and test_system_prompt and api_source:
                    if test_model not in available_models:
                        available_models[test_model] = []
                    available_models[test_model].append({
                        "prompt": test_system_prompt, 
                        "api": api_source,
                        "comments": comments
                    })
        if not available_models:
            raise Exception("No valid models found in CSV")
    except Exception as e:
        print(f"Error: Could not load test model and system prompt from CSV. Loading default model with default system prompt. Error: {str(e)}")
        available_models[DEFAULT_MODEL] = [{
            "prompt": DEFAULT_SYSTEM_PROMPT, 
            "api": DEFAULT_API,
            "comments": ""
        }]
    return available_models

def set_print_responses():
    global print_responses
    choice = input("Would you like to print API response transcripts? (y/n): ").lower()
    print_responses = choice == 'y'

def load_test_cases() -> List[TestCase]:
    test_cases = []
    test_cases_dir = "web_search/test_cases"
    list_of_files = glob.glob(f"{test_cases_dir}/web_search_test_cases_*.csv")
    if not list_of_files:
        raise FileNotFoundError("No test cases files found in the specified directory.")
    latest_file = max(list_of_files, key=os.path.getctime)
    
    with open(latest_file, "r", newline='') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            try:
                web_search_expected = row.get("web_search_expected", "").lower() == "true"
                score = float(row.get("Score", 0.0))
                test_question = row["triggering_user_message"].strip()
                context = row.get("context", "")
                test_cases.append(TestCase(
                    web_search_expected=web_search_expected,
                    score=score,
                    test_question=test_question,
                    context=context
                ))
            except Exception as e:
                print(f"Error parsing row: {row}")
                print(f"Error details: {str(e)}")
    
    return test_cases

def select_models(available_models: Dict[str, List[Dict[str, str]]]) -> List[Tuple[str, Dict[str, str]]]:
    global print_responses
    print(f"\nAPI response transcripts will {'' if print_responses else 'not '}be printed. To toggle this setting, type 'print'.")
    print("Currently loading the most recent test_cases file. To load different test_cases, type 'load {file path}'")
    
    model_prompt_combinations = []
    for model, prompts in available_models.items():
        for i, prompt_dict in enumerate(prompts):
            api = prompt_dict["api"]
            comments = prompt_dict["comments"]
            comment_str = f" - {comments}" if comments else ""
            model_prompt_combinations.append((model, prompt_dict))
            print(f"{len(model_prompt_combinations)}. {model} ({api}) - Prompt {i+1} {comment_str}")

    while True:
        try:
            choice = input("Enter the number(s) of the model-prompt combination(s) you want to test (comma-separated), \n'all' for all combinations, or 'load' to change test cases: ")
            if choice.lower() == 'print':
                print_responses = not print_responses
                print(f"API responses will {'' if print_responses else 'not '}be printed.")
                continue
            elif choice.lower().startswith('load '):
                new_file_path = choice.split(' ', 1)[1]
                try:
                    global load_test_cases
                    def load_test_cases():
                        return load_specific_test_cases(new_file_path)
                    print(f"Test cases will now be loaded from: {new_file_path}")
                    continue
                except Exception as e:
                    print(f"Error loading file: {str(e)}")
                    continue
            elif choice.lower() == 'all':
                return model_prompt_combinations
            
            selected_indices = [int(idx.strip()) - 1 for idx in choice.split(',')]
            selected_combinations = [model_prompt_combinations[i] for i in selected_indices if 0 <= i < len(model_prompt_combinations)]
            
            if not selected_combinations:
                print("No valid model-prompt combinations selected. Please try again.")
            else:
                return selected_combinations
        except ValueError:
            print("Invalid input. Please enter number(s), 'all', 'load {file path}', or 'print'.")

def load_specific_test_cases(file_path: str) -> List[TestCase]:
    test_cases = []
    with open(file_path, "r", newline='') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            test_cases.append(TestCase(
                test_question=row["test_question"],
                web_search_expected=row.get("web_search_expected", "").lower() == "true"
            ))
    return test_cases

def parse_context(context: str) -> List[Dict[str, str]]:
    messages = []
    if context:
        context_messages = context.split("MESSAGE:")
        for msg in context_messages:
            if msg.strip():
                parts = msg.strip().split(" ", 1)
                if len(parts) == 2:
                    role, content = parts
                    # Map roles to valid API roles
                    if role.upper() == "USER":
                        api_role = "user"
                    elif role.upper() in ["AGENT", "ASSISTANT", "AI"]:
                        api_role = "assistant"
                    else:
                        continue  # Skip invalid roles
                    messages.append({"role": api_role, "content": content.strip()})
    return messages

async def get_completion_openai(client: OpenAI, model: str, system_prompt: str, test_question: str, context: str, steering_config: Dict[str, Any]) -> Any:
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    # Parse and add context messages
    messages.extend(parse_context(context))
    
    messages.append({"role": "user", "content": test_question})
    
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=model,
        messages=messages,
        tools=[web_search_tool],
        extra_headers={"anthropic-beta": "steering-2024-06-04"},
        extra_body={"steering": steering_config}
    )
    if print_responses:
        print("OpenAI API response:", response)
    return response

async def get_completion_anthropic(client: anthropic.Anthropic, model: str, system_prompt: str, test_question: str, context: str, steering_config: Dict[str, Any]) -> Any:
    messages = parse_context(context)
    
    messages.append({"role": "user", "content": test_question})
    
    response = await asyncio.to_thread(
        client.messages.create,
        model=model,
        max_tokens=1024,
        temperature=1,
        system=system_prompt,
        tools=[web_search_tool_anthropic],
        messages=messages,
        extra_headers={"anthropic-beta": "steering-2024-06-04"},
        extra_body={"steering": steering_config}
    )
    
    if print_responses:
        print("Anthropic API response:", response)
    return response

def evaluate_web_search(response: Any, api_source: str) -> bool:
    web_search_performed = False
    if api_source == "OpenAI":
        for choice in response.choices:
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    if tool_call.function.name == 'web_search':
                        web_search_performed = True
                        break
                if web_search_performed:
                    break
    elif api_source == "Anthropic":
        for block in response.content:
            if isinstance(block, anthropic.types.ContentBlock) and block.type == 'tool_calls':
                for tool_call in block.tool_calls:
                    if tool_call.type == 'function' and tool_call.function.name == 'web_search':
                        web_search_performed = True
                        break
                if web_search_performed:
                    break
    return web_search_performed

async def process_test_case(openai_client: OpenAI, anthropic_client: anthropic.Anthropic, 
                            test_model: str, test_system_prompt: str, api_source: str, 
                            test_case: TestCase, steering_config: Dict[str, Any]) -> Tuple[bool, bool, str, float, str, str]:
    try:
        if api_source == "OpenAI":
            response = await get_completion_openai(openai_client, test_model, test_system_prompt, test_case.test_question, test_case.context, steering_config)
            message_text = response.choices[0].message.content if response.choices else ""
        elif api_source == "Anthropic":
            response = await get_completion_anthropic(anthropic_client, test_model, test_system_prompt, test_case.test_question, test_case.context, steering_config)
            message_text = "".join(block.text for block in response.content if block.type == 'text')
        else:
            raise ValueError(f"Unsupported API source: {api_source}")

        web_search_performed = evaluate_web_search(response, api_source)
        search_query = extract_search_query(response, api_source)
        result = get_result(web_search_performed, test_case.web_search_expected)
        
        if print_responses:
            print(f"API Response (Message Text): {message_text}")
            print(f"Search Query: {search_query}")
            print(f"{result}, {test_case.score:.2f}, {test_case.web_search_expected}, {web_search_performed}, \"{test_case.test_question}\", \"{test_case.context}\"")
        
        return web_search_performed, test_case.web_search_expected, test_case.test_question, test_case.score, message_text, search_query
    except Exception as e:
        print(f"Error processing question: {test_case.test_question}")
        print(f"Error details: {str(e)}")
        print(f"Context: {test_case.context}")
        return False, test_case.web_search_expected, test_case.test_question, 0.0, "", ""

def extract_search_query(response: Any, api_source: str) -> str:
    if api_source == "OpenAI":
        for choice in response.choices:
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    if tool_call.function.name == 'web_search':
                        return json.loads(tool_call.function.arguments).get('query', '')
    elif api_source == "Anthropic":
        for block in response.content:
            if block.type == 'tool_use' and block.name == 'web_search':
                return block.input.get('query', '')
    return ""

def get_result(performed: bool, expected: bool) -> str:
    if performed and expected:
        return "TP"
    elif performed and not expected:
        return "FP"
    elif not performed and not expected:
        return "TN"
    else:
        return "FN"

async def web_search_unit_test(openai_client: OpenAI, anthropic_client: anthropic.Anthropic, 
                               test_model: str, test_system_prompt: str, api_source: str, 
                               comments: str, steering_config: Dict[str, Any]) -> Tuple[DetailedTestResult, List[Tuple[bool, bool, str, float, str, str]]]:
    test_cases = load_test_cases()
    tasks = [process_test_case(openai_client, anthropic_client, test_model, test_system_prompt, api_source, test_case, steering_config)
             for test_case in test_cases]
    results = await asyncio.gather(*tasks)

    score_level_stats = {score: {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "total": 0} for score in range(-5, 6)}
    
    for performed, expected, _, score, _, _ in results:
        rounded_score = round(score)
        result = get_result(performed, expected)
        score_level_stats[rounded_score][result] += 1
        score_level_stats[rounded_score]["total"] += 1

    true_positives = sum(stats["TP"] for stats in score_level_stats.values())
    false_positives = sum(stats["FP"] for stats in score_level_stats.values())
    true_negatives = sum(stats["TN"] for stats in score_level_stats.values())
    false_negatives = sum(stats["FN"] for stats in score_level_stats.values())
    
    total_cases = len(test_cases)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    success_rate = (true_positives + true_negatives) / total_cases

    for score, stats in score_level_stats.items():
        tp, fp, tn, fn = stats['TP'], stats['FP'], stats['TN'], stats['FN']
        stats['precision'], stats['recall'], stats['success_rate'] = calculate_stats(tp, fp, tn, fn)

    detailed_result = DetailedTestResult(
        model=test_model,
        api_source=api_source,
        precision=precision,
        recall=recall,
        success_rate=success_rate,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        total_cases=total_cases,
        comments=comments,
        score_level_stats=score_level_stats
    )
    
    print_score_level_stats(score_level_stats)
    write_results_to_csv(results, detailed_result, steering_config)

    return detailed_result, results

def calculate_stats(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    success_rate = (tp + tn) / total if total > 0 else 0
    return precision, recall, success_rate

def print_score_level_stats(score_level_stats):
    print("\nScore Level Statistics:")
    print("Score | Total | TP | FP | TN | FN | Precision | Recall | Success Rate")
    print("-" * 70)
    for score in range(-5, 6):
        stats = score_level_stats[score]
        tp, fp, tn, fn = stats['TP'], stats['FP'], stats['TN'], stats['FN']
        precision, recall, success_rate = calculate_stats(tp, fp, tn, fn)
        print(f"{score:5d} | {stats['total']:5d} | {tp:2d} | {fp:2d} | {tn:2d} | {fn:2d} | {precision:9.2%} | {recall:6.2%} | {success_rate:11.2%}")

def write_results_to_csv(results, detailed_result: DetailedTestResult, steering_config: Dict[str, Any]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"web_search/results/web_search_results_{timestamp}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        
        # Write the results log at the top of the CSV
        csvwriter.writerow(["Web search unit test results"])
        csvwriter.writerow([f"Model: {detailed_result.model}"])
        csvwriter.writerow([f"API Source: {detailed_result.api_source}"])
        
        # Add the web search tool description
        web_search_description = web_search_tool["function"]["description"] if detailed_result.api_source == "OpenAI" else web_search_tool_anthropic["description"]
        csvwriter.writerow(["Web Search Tool Description:"])
        # Split the description into multiple rows if it's too long
        for line in web_search_description.split(". "):
            csvwriter.writerow([line.strip() + "."])
        
        # Add steering configuration
        csvwriter.writerow(["Steering Configuration:"])
        for feature, level in steering_config.get("feature_levels", {}).items():
            csvwriter.writerow([f"{feature}: {level}"])
        
        csvwriter.writerow([f"\nWeb search unit test results for model {detailed_result.model} ({detailed_result.api_source}):"])
        csvwriter.writerow([f"Comments: {detailed_result.comments}"])
        csvwriter.writerow([f"Overall Precision: {detailed_result.precision:.2%} ({detailed_result.true_positives}/{detailed_result.true_positives + detailed_result.false_positives})"])
        csvwriter.writerow([f"Overall Recall: {detailed_result.recall:.2%} ({detailed_result.true_positives}/{detailed_result.true_positives + detailed_result.false_negatives})"])
        csvwriter.writerow([f"Overall Success rate: {detailed_result.success_rate:.2%} ({detailed_result.true_positives + detailed_result.true_negatives}/{detailed_result.total_cases})"])
        csvwriter.writerow([f"Results written to {filename}"])
        csvwriter.writerow([])  # Empty row for separation
        
        # Update the score level statistics table
        csvwriter.writerow(["Score Level Statistics"])
        csvwriter.writerow(["Score", "Total", "TP", "FP", "TN", "FN", "Precision", "Recall", "Success Rate"])
        for score in range(-5, 6):
            stats = detailed_result.score_level_stats[score]
            tp, fp, tn, fn = stats['TP'], stats['FP'], stats['TN'], stats['FN']
            precision, recall, success_rate = calculate_stats(tp, fp, tn, fn)
            csvwriter.writerow([
                score,
                stats['total'],
                tp,
                fp,
                tn,
                fn,
                f"{precision:.2%}",
                f"{recall:.2%}",
                f"{success_rate:.2%}"
            ])

        csvwriter.writerow([])  # Empty row for separation

        # Write the column headers for individual results
        csvwriter.writerow(["result", "score", "web_search_expected", "web_search_performed", "test_question", "api_response", "search_query", "context", "model", "api_source", "comments"])
        
        # Write the individual results
        for performed, expected, question, score, api_response, search_query in results:
            result = get_result(performed, expected)
            test_case = next((tc for tc in load_test_cases() if tc.test_question == question), None)
            if test_case:
                csvwriter.writerow([
                    result, f"{score:.2f}", expected, performed, question, 
                    api_response, search_query, test_case.context, 
                    detailed_result.model, detailed_result.api_source, 
                    detailed_result.comments
                ])
            else:
                print(f"Warning: Test case not found for question: {question}")

    print(f"Results written to {filename}")

async def main():
    set_print_responses()
    available_models = load_available_models()

    while True:
        selected_combinations = select_models(available_models)

        # Ask for steering configuration
        steering_config = {}
        use_steering = input("Do you want to use feature steering? (y/n): ").lower() == 'y'
        if use_steering:
            while True:
                feature_id = input("Enter the feature ID (or 'done' to finish): ")
                if feature_id.lower() == 'done':
                    break
                try:
                    level = float(input("Enter the steering level (-10 to 10): "))
                    if -10 <= level <= 10:
                        steering_config[f"feat_34M_20240604_{feature_id}"] = level
                    else:
                        print("Invalid level. Please enter a value between -10 and 10.")
                except ValueError:
                    print("Invalid input. Please enter a numeric value for the steering level.")

        for test_model, prompt_dict in selected_combinations:
            test_system_prompt = prompt_dict["prompt"]
            api_source = prompt_dict["api"]
            comments = prompt_dict["comments"]
            if api_source == "OpenAI" and openai_client is None:
                print(f"Skipping test for {test_model} as OpenAI client is not initialized.")
                continue
            if api_source == "Anthropic" and anthropic_client is None:
                print(f"Skipping test for {test_model} as Anthropic client is not initialized.")
                continue
            
            detailed_result, results = await web_search_unit_test(
                openai_client, anthropic_client, test_model, test_system_prompt, 
                api_source, comments, {"feature_levels": steering_config} if use_steering else {}
            )

            # Print the detailed results
            print(f"\nWeb search unit test results for model {detailed_result.model} ({detailed_result.api_source}):")
            print(f"Comments: {detailed_result.comments}")
            print(f"Precision: {detailed_result.precision:.2%} ({detailed_result.true_positives}/{detailed_result.true_positives + detailed_result.false_positives})")
            print(f"Recall: {detailed_result.recall:.2%} ({detailed_result.true_positives}/{detailed_result.true_positives + detailed_result.false_negatives})")
            print(f"Success rate: {detailed_result.success_rate:.2%} ({detailed_result.true_positives + detailed_result.true_negatives}/{detailed_result.total_cases})")

            fetch_samples = input("Would you like to fetch sample results? (y/n): ").lower()
            if fetch_samples == 'y':
                fetch_sample_results(results, detailed_result.score_level_stats)

        run_again = input("\nWould you like to run another test with different models or steering configuration? (y/n): ").lower()
        if run_again != 'y':
            print("Thank you for using the web search unit test tool. Goodbye!")
            break

if __name__ == "__main__":
    asyncio.run(main())