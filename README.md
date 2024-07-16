# Web Search Unit Test Tool
This is repo is an open source LLM web search unit test. This was originally made at Hume AI, and all Hume-related identifying information redacted and replaced.

## Overview
This tool is designed to evaluate the performance of various LLMs in determining when to perform web searches based on user questions. It currently supports testing models from both OpenAI and Anthropic APIs. The main objectives are:

1. Run web search on a series of 100+ high-quality test cases and generate metrics like precision & recall.
2. Improve precision/recall web search by at least 10 percentage points through modifications to the prompt, tool description, and/or other variables.
3. Create an extensible framework that can be built upon for other evaluations and continued improvement of web search evaluations.

## How It Works

The web search unit test function follows these steps:

1. Loads a dataset of user questions from real conversational logs.
2. Sends these questions to OpenAI/Anthropic API (configurable).
3. Checks if a web search query has been performed in response to each question.
4. Reports the percentage of user questions which resulted in a successful web search query, along with other relevant metrics.

This project is structured into two main components:

1. `chat_ranker`: Converts raw user chat log data into a CSV with LLM scores of various criteria.
2. `web_search`: Conducts unit tests given different models, system prompts, and scoring criteria.

## How to Use

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/menhguin/llm-benchmarks.git
   cd llm-benchmarks
   ```

2. Install the required dependencies (mainly just OpenAI, Anthropic and python-dotenv):
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   Create a `.env` file in the root directory and add your OpenAI and Anthropic API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

### Running the Test

1. To run the web search unit test:
   ```
   python web_search/test_web_search.py
   ```

2. To add a new dataset:
   - Add a new Json with the same format as the current one. Currently the filepath just detects whatever file in the project folder is named `live_test_cases.json`. Note that this unit test natively supports multi-turn message log input via both Anthropic and OpenAI APIs.
   - Run `python chat_ranker/test_case_ranker.py`.

3. To add a new model or system prompt:
   - Edit `web_search/test_system_prompts.csv`.
   - Run `python web_search/test_web_search.py` and select the new model/prompt when prompted.

## Field Descriptions

### Input CSV Fields

1. `web_search_expected` (boolean): Indicates whether a web search is expected for this test case.
2. `follow_up_expected` (boolean): Indicates whether a follow-up question is expected for this test case.
3. `Score` (float): A rating between -5 and 5 indicating the likelihood of triggering a web search.
4. `triggering_user_message` (string): The user's message being evaluated for potential web search triggering.
5. `context` (string): Previous conversation context, if any.
6. `function_call_text` (string): The specific text of the function call, if applicable.

### Output CSV Fields

1. `result` (string): Outcome of the test case (TP, FP, TN, FN).
2. `score` (float): Same score as in the input CSV.
3. `web_search_expected` (boolean): Whether a web search was expected.
4. `web_search_performed` (boolean): Whether a web search was actually performed.
5. `test_question` (string): The user's message that was tested.
6. `api_response` (string): The response received from the API.
7. `search_query` (string): The actual search query used if a web search was performed.
8. `context` (string): The conversation context provided.
9. `model` (string): The name of the model used for testing.
10. `api_source` (string): The API provider used.
11. `comments` (string): Any additional comments or notes about the test case.

## Todo List:

- **Follow-up questions** - Certain queries involve/require follow-up questions. Currently I am trying to figure out how to account for this. It’s important-ish for max granularity because recent system prompts are more likely to ask follow-up but this gets them marked as less precision.
- **Performance and modularisation** - With the newest addition of follow-up questions, the eval takes about 2 minutes. This is substantially slower than the ~10 seconds it used to be (and the ranker is still about 10 seconds), so I suspect I need to speed some parts up.
- **Manual review of test cases -** affects edge case accuracy of eval.
- **Refactoring for better handling of column names for future evals:** Currently the column/entry fields are hardcoded to suit the web search eval. It actually works fine if you just change the system prompt and keep the column names the same/change all the column names throughout, but someone should probably make those dynamic/input-based eventually.
- **Figure out how to get real false negatives**
- **Include other fields:** Filter by time/model, lag time, emotional data, qualitatively parsing actual query results
- **Comparison of different settings/prompts and listing examples where the different settings resulted in different responses**
- **Documentation**
