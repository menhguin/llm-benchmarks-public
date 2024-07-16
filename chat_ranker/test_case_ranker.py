import anthropic
from openai import AsyncOpenAI
from dotenv import load_dotenv
import csv
import os
import sys
import json
import asyncio
from typing import List, Dict, Tuple, Any
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Global variable for print responses
print_responses = True

# Initialize API clients with better error handling
async def initialize_clients():
    try:
        openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    except KeyError:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please make sure you have set the OPENAI_API_KEY in your .env file or environment.")
        sys.exit(1)

    try:
        anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except KeyError:
        print("Warning: ANTHROPIC_API_KEY not found in environment variables.")
        print("Anthropic models will not be available for testing.")
        anthropic_client = None
    except Exception as e:
        print(f"Error initializing Anthropic client: {str(e)}")
        anthropic_client = None

    return openai_client, anthropic_client

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_SYSTEM_PROMPT = "Analyze user messages for web search relevance."
DEFAULT_API = "OpenAI"

def get_available_projects() -> List[str]:
    return [folder for folder in os.listdir() if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "test_system_prompts.csv"))]

def select_project(projects: List[str]) -> str:
    print("Available projects:")
    for i, project in enumerate(projects, 1):
        print(f"{i}. {project}")
    
    while True:
        try:
            choice = input("Enter the destination folder for output test cases: ")
            index = int(choice) - 1
            if 0 <= index < len(projects):
                return projects[index]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

async def load_available_models() -> Dict[str, Dict[str, str]]:
    available_models = {}
    try:
        with open("chat_ranker/test_system_prompts.csv", "r", newline='') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                test_model = row.get("test_model", "").strip()
                test_system_prompt = row.get("test_system_prompt", "").strip()
                api_source = row.get("api_source", "").strip()
                comments = row.get("comments", "").strip()
                if test_model and test_system_prompt and api_source:
                    available_models[test_model] = {
                        "prompt": test_system_prompt, 
                        "api": api_source,
                        "comments": comments
                    }
        if not available_models:
            raise Exception("No valid models found in CSV")
    except Exception as e:
        print(f"Error: Could not load test model and system prompt from CSV. Loading default model with default system prompt. Error: {str(e)}")
        available_models[DEFAULT_MODEL] = {
            "prompt": DEFAULT_SYSTEM_PROMPT, 
            "api": DEFAULT_API,
            "comments": ""
        }
    return available_models

def set_print_responses():
    global print_responses
    choice = input("Would you like to print API response transcripts? (y/n): ").lower()
    print_responses = choice == 'y'

async def load_queries(project: str) -> List[Dict]:
    queries = []
    try:
        with open(f"{project}/live_test_cases.json", "r") as f:
            data = json.load(f)
            
            total_examples = len(data)
            filtered_data = []
            
            no_user_message = 0
            baymax_health_database = 0
            valid_queries = 0
            
            for row in data:
                triggering_user_message = row.get('triggering_user_message', '')
                prior_messages = row.get('prior_messages', [])
                
                # Check for Baymax fetching info from the health database. You can delete this, this is just to show filtering examples.
                prior_messages_text = ' '.join(str(msg.get('message_text', '')) for msg in prior_messages)
                if "baymax" in prior_messages_text.lower() and "health database" in prior_messages_text.lower():
                    baymax_health_database += 1
                    continue
                
                if not triggering_user_message:
                    no_user_message += 1
                    continue
                
                # Filter out SYSTEM_PROMPT messages and create context
                context = "\n".join([
                    f"{msg['event_type']}: {msg.get('message_text', '')}"
                    for msg in prior_messages
                    if msg['event_type'] != 'SYSTEM_PROMPT' and msg.get('message_text')
                ])
                
                filtered_data.append({
                    "user_message": triggering_user_message,
                    "context": context,
                    "function_call_text": row.get('function_call_text', '')
                })
                valid_queries += 1
            
            queries = filtered_data
            
            print(f"Total examples in JSON: {total_examples}")
            print(f"Examples with no user message: {no_user_message}")
            print(f"Examples with 'Baymax' and 'health database': {baymax_health_database}")
            print(f"Valid queries after filtering: {valid_queries}")
        
        if not queries:
            raise Exception("No valid queries found in JSON after filtering")
    except Exception as e:
        print(f"Error: Could not load queries from JSON. Error: {str(e)}")
        print("Fallback to default query")
        queries = [{"user_message": "Default query for testing", "context": "", "function_call_text": ""}]  # Fallback to a default query
    
    return queries

def inspect_json(project: str, num_entries: int = 5):
    try:
        with open(f"{project}/live_test_cases.json", "r") as f:
            data = json.load(f)
        
        print(f"\nInspecting first {num_entries} entries of the JSON file:")
        for i, entry in enumerate(data[:num_entries], 1):
            print(f"\nEntry {i}:")
            print(f"Triggering user message: {entry.get('triggering_user_message', 'Not found')}")
            print("Prior messages:")
            for msg in entry.get('prior_messages', [])[:5]:  # Show first 5 prior messages
                print(f"  - {msg.get('event_type')}: {msg.get('message_text', '')[:50]}...")
            print("..." if len(entry.get('prior_messages', [])) > 5 else "")
            print(f"Function call text: {entry.get('function_call_text', 'Not found')[:100]}...")
    except Exception as e:
        print(f"Error inspecting JSON: {str(e)}")

def select_models(available_models: Dict[str, Dict[str, str]]) -> List[str]:
    global print_responses
    print(f"\nAPI response transcripts will {'' if print_responses else 'not '}be printed. To toggle this setting, type 'print'.")
    models = list(available_models.keys())
    print("Available models:")
    for i, model in enumerate(models, 1):
        api = available_models[model]["api"]
        comments = available_models[model]["comments"]
        comment_str = f" - {comments}" if comments else ""
        print(f"{i}. {model} ({api}){comment_str}")

    while True:
        try:
            choice = input("Enter the number(s) of the model(s) you want to test (comma-separated), \n'all' for all models: ")
            if choice.lower() == 'print':
                print_responses = not print_responses
                print(f"API responses will {'' if print_responses else 'not '}be printed.")
                continue
            elif choice.lower() == 'all':
                return models
            
            selected_indices = [int(idx.strip()) - 1 for idx in choice.split(',')]
            selected_models = [models[i] for i in selected_indices if 0 <= i < len(models)]
            
            if not selected_models:
                print("No valid models selected. Please try again.")
            else:
                return selected_models
        except ValueError:
            print("Invalid input. Please enter number(s), 'all', or 'print'.")

def parse_context(context: str) -> List[Dict[str, str]]:
    messages = []
    if context:
        context_messages = context.split("\n")
        last_role = None
        for msg in context_messages:
            if msg.strip():
                parts = msg.strip().split(":", 1)
                if len(parts) == 2:
                    role, content = parts
                    if role.upper() == "USER_MESSAGE":
                        api_role = "user"
                    elif role.upper() == "AGENT_MESSAGE":
                        api_role = "assistant"
                    else:
                        continue  # Skip invalid roles
                    
                    # If the current role is the same as the last one, combine the messages
                    if api_role == last_role and messages:
                        messages[-1]["content"] += f" {content.strip()}"
                    else:
                        messages.append({"role": api_role, "content": content.strip()})
                        last_role = api_role
    return messages

async def get_completion_openai(client, model: str, system_prompt: str, user_message: str, context: str) -> Any:
    formatted_system_prompt = system_prompt.format(context=context, query=user_message)
    messages = [
        {"role": "system", "content": formatted_system_prompt},
    ]
    
    context_messages = parse_context(context)
    
    # Ensure the messages alternate correctly
    if context_messages:
        if context_messages[0]["role"] == "assistant":
            context_messages.insert(0, {"role": "user", "content": "Start of conversation"})
        messages.extend(context_messages)
    
    # Add the current user message
    if messages[-1]["role"] == "user":
        messages[-1]["content"] += f" {user_message}"
    else:
        messages.append({"role": "user", "content": user_message})
    
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response

async def get_completion_anthropic(client, model: str, system_prompt: str, user_message: str, context: str) -> Any:
    formatted_system_prompt = system_prompt.format(context=context, query=user_message)
    
    context_messages = parse_context(context)
    
    # Ensure the messages alternate correctly
    if context_messages:
        if context_messages[0]["role"] == "assistant":
            context_messages.insert(0, {"role": "user", "content": "Start of conversation"})
    
    # Add the current user message
    if context_messages and context_messages[-1]["role"] == "user":
        context_messages[-1]["content"] += f" {user_message}"
    else:
        context_messages.append({"role": "user", "content": user_message})
    
    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=formatted_system_prompt,
        messages=context_messages,
    )
    return response

def extract_score_and_expectations(content: str) -> Tuple[float, bool, bool]:
    lines = content.split('\n')
    score = 0
    web_search_expected = False
    follow_up_expected = False
    for line in lines:
        if line.lower().startswith("web search expected:"):
            web_search_expected = line.split(':')[1].strip().lower() == 'true'
        elif line.lower().startswith("follow-up expected:"):
            follow_up_expected = line.split(':')[1].strip().lower() == 'true'
        elif line.lower().startswith("rating:") or line.lower().startswith("score:"):
            try:
                score = float(line.split(':')[1].strip().split()[0])
            except ValueError:
                score = 0
    return score, web_search_expected, follow_up_expected

async def process_query(openai_client, anthropic_client, test_model: str, test_system_prompt: str, api_source: str, query: Dict) -> Tuple[str, float, bool, bool, str, str]:
    try:
        user_message = query["user_message"]
        context = query["context"]
        function_call_text = query["function_call_text"]
        
        if api_source == "OpenAI":
            response = await get_completion_openai(openai_client, test_model, test_system_prompt, user_message, context)
            content = response.choices[0].message.content
        elif api_source == "Anthropic":
            response = await get_completion_anthropic(anthropic_client, test_model, test_system_prompt, user_message, context)
            content = response.content[0].text
        else:
            raise ValueError(f"Unsupported API source: {api_source}")

        score, web_search_expected, follow_up_expected = extract_score_and_expectations(content)
        
        if score == 0 and not web_search_expected and not follow_up_expected:
            print(f"Warning: Unexpected result for query: {user_message}")
            print(f"API Response: {content}")
        
        return user_message, score, web_search_expected, follow_up_expected, context, function_call_text
    except Exception as e:
        print(f"Error processing query: {user_message}")
        print(f"Error details: {str(e)}")
        return user_message, 0, False, False, context, ""

async def analyze_queries(openai_client, anthropic_client, test_model: str, test_system_prompt: str, api_source: str, queries: List[Dict]) -> List[Tuple[str, float, bool, bool, str, str]]:
    tasks = [process_query(openai_client, anthropic_client, test_model, test_system_prompt, api_source, query) for query in queries]
    scored_queries = await asyncio.gather(*tasks)
    
    # Sort queries by score in descending order
    scored_queries.sort(key=lambda x: x[1], reverse=True)

    return scored_queries

def save_to_csv(scored_queries: List[Tuple[str, float, bool, bool, str, str]], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['web_search_expected', 'follow_up_expected', 'Score', 'triggering_user_message', 'context', 'function_call_text'])
        for query, score, web_search_expected, follow_up_expected, context, function_call_text in scored_queries:
            writer.writerow([web_search_expected, follow_up_expected, f"{score:.2f}", query, context, function_call_text])
    print(f"Results saved to {output_file}")

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

async def main():
    projects = get_available_projects()
    selected_project = select_project(projects)
    
    set_print_responses()
    available_models = await load_available_models()
    selected_models = select_models(available_models)

    # Load queries from JSON
    queries = await load_queries(selected_project)

    openai_client, anthropic_client = await initialize_clients()

    timestamp = get_timestamp()

    for test_model in selected_models:
        test_system_prompt = available_models[test_model]["prompt"]
        api_source = available_models[test_model]["api"]
        
        if api_source == "OpenAI" and openai_client is None:
            print(f"Skipping test for {test_model} as OpenAI client is not initialized.")
            continue
        if api_source == "Anthropic" and anthropic_client is None:
            print(f"Skipping test for {test_model} as Anthropic client is not initialized.")
            continue

        print(f"\nAnalyzing queries using model {test_model} ({api_source}):")
        scored_queries = await analyze_queries(openai_client, anthropic_client, test_model, test_system_prompt, api_source, queries)

        # Save results to CSV with timestamp and project name
        output_file = f"{selected_project}/test_cases/{selected_project}_test_cases_{timestamp}.csv"
        save_to_csv(scored_queries, output_file)

        print("\nAll queries sorted by score (highest to lowest):")
        for query, score, web_search_expected, follow_up_expected, context, function_call_text in scored_queries:
            print(f"Web_search_expected: {web_search_expected}, Follow_up_expected: {follow_up_expected}, Score: {score:.2f}, Triggering User Message: {query}")
            print(f"Context:\n{context}")
            print(f"Function call text: {function_call_text}\n")

if __name__ == "__main__":
    asyncio.run(main())