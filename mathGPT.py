import openai
import ast
import math
import re
import subprocess
import os
import numpy as np

# Set up your OpenAI API credentials
openai.api_key = API_KEY

# Define the function to interact with ChatGPT


def chat_with_gpt(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        temperature=0.4,
        n=6,
        stop=None,
    )
    return response.choices


def evaluate_functions(generated_functions, expression, input_values):
    function_stats_accuracy = []
    valid_generated_functions = []
    for i, function in enumerate(generated_functions, start=1):
        print(f"Evaluating Function {i}:\n")
        max_percent_error = 0

        try:
            # Transform C-style code to Python-style code
            python_code = re.sub(r'int\s+compute_result',
                                 'def compute_result', function)
            python_code = re.sub(r'int\s+', '', python_code)
            python_code = re.sub(r'{', ':', python_code)
            python_code = re.sub(r'}', '', python_code)
            python_code = re.sub(r';', '', python_code)
            python_code = re.sub(r'result\s*;', 'result = 0', python_code)

            # Check if the generated function is valid
            try:
                ast.parse(python_code)
                valid_generated_functions.append(function)
            except SyntaxError as e:
                print(f"Error: Generated function is invalid: {e}")
                continue

            # Compile the transformed Python code
            try:
                function_ast = ast.parse(python_code)
                function_body = function_ast.body[0]
                function_code = compile(function_ast, filename='', mode='exec')
                local_vars = {}
                global_vars = {'math': math, 'sqrt': math.sqrt}
                exec(function_code, global_vars, local_vars)

                # Get the function object from the compiled code
                function_obj = local_vars[function_body.name]
            except Exception as e:
                print(f"Error compiling function: {e}")
                continue

            for input_value in input_values:
                try:
                    # Evaluate the function with the input value and return the result
                    output = function_obj(input_value)

                    # Calculate the expected output by plugging the input value into the math expression
                    expected_output = evaluate_math_expression(
                        expression, input_value)

                    # Calculate the percent error
                    percent_error = abs(
                        (output - expected_output) / expected_output) * 100

                    # Update the maximum percent error if needed
                    max_percent_error = max(max_percent_error, percent_error)

                except NameError as e:
                    if "name 'result' is not defined" in str(e):
                        print(f"Error: Generated function is invalid: {e}")
                        valid_generated_functions.remove(function)
                        break
                    else:
                        print(
                            f"Error evaluating function with input {input_value}: {e}")
                        break
                except Exception as e:
                    print(
                        f"Error evaluating function with input {input_value}: {e}")
                    break

            else:
                # Print the maximum percent error for the function
                print(f"Max Percent Error: {max_percent_error}%\n")

                # Add the function and its stats to the list
                function_stats_accuracy.append({
                    "function": function,
                    "stats": {
                        "Max Percent Error": max_percent_error,
                    }
                })

        except Exception as e:
            print(f"Error compiling function: {e}")
    return valid_generated_functions, function_stats_accuracy


# Implement the evaluate_math_expression function
def evaluate_math_expression(expression, input_value):
    # Create a dictionary containing all the functions and constants from the math module
    math_namespace = {name: getattr(math, name)
                      for name in dir(math) if not name.startswith("_")}

    # Add the input value to the dictionary
    math_namespace['x'] = input_value

    # Evaluate the expression using the math namespace
    result = eval(expression, math_namespace)
    return result

# Function to parse Klee statistics


def parse_klee_stats(stats_output):
    lines = stats_output.split("\n")
    header_line = lines[1]
    data_line = lines[3]

    headers = [header.strip() for header in header_line.split("|")[1:-1]]
    data_values = [data.strip() for data in data_line.split("|")[1:-1]]

    stats = dict(zip(headers, data_values))
    return stats


# Step 1: Receive the mathematical expression from the user
expression = input("Enter a mathematical expression: ")

# Step 2: Prepare the input for ChatGPT
prompt = f"Given the expression '{expression}', please generate an integer-based function in C with the name compute_result that computes the result without using external function (like sqrt), without floating-point values, and without using iterations (like while and for loop). The function should take the same parameters but in integer-based form and return an integer result and have the highest approximation result compared to the original expression and please ouput exact like this format:\n\nint compute_result(int x) {{\n *your code goes here*\n}} (remember the open bracket is at same line as function name do not break line for open bracket."

# Step 3: Interact with ChatGPT
response_choices = chat_with_gpt(prompt)

# Step 4: Parse and process the response choices
generated_functions = [choice.text.strip() for choice in response_choices]

# Remove duplicate functions
generated_functions = list(set(generated_functions))

# Define a list of disallowed keywords
disallowed_keywords = ["sqrt", "for", "while", "pow", "floor"]

# Check if any of the generated functions contain the sqrt function
for i, function in enumerate(generated_functions):
    # Check if a floating-point value is present in the function
    has_floating_point_value = re.search(r"\d+\.\d+", function) is not None

    if any(keyword in function for keyword in disallowed_keywords) or has_floating_point_value:
        # Generate a new prompt to request a corrected function
        new_prompt = f"The following generated function contains disallowed keywords: {function}. Please generate a corrected version of this function that does not use external function (like sqrt), do not use floating-point values, and do not use iterations (like while and for loop). The function should take the same parameters but in integer-based form and return an integer result and have the highest approximation result compared to the original expression and please ouput exact like this format:\n\nint compute_result(int x) {{\n *your code goes here*\n}} (remember the open bracket is at same line as function name do not break line for open bracket."

        new_response_choices = chat_with_gpt(new_prompt)
        new_function = new_response_choices[0].text.strip()
        generated_functions[i] = new_function

# Print the generated functions
for i, function in enumerate(generated_functions, start=1):
    print(f"Generated function {i}:\n{function}")

# Step 5: Generate input values
# Example input values, adjust as needed
start = 0
end = 10
input_values = list(range(start, end + 1))

# Step 6: Evaluate generated functions and get function stats
valid_generated_functions, function_stats_accuracy = evaluate_functions(
    generated_functions, expression, input_values)

# Print the generated functions
for i, valid_function in enumerate(valid_generated_functions, start=1):
    print(f"Valid function {i}:\n{valid_function}")

# Sort the function stats based on the maximum percent error (accuracy)
function_stats_accuracy = sorted(
    function_stats_accuracy, key=lambda x: x["stats"]["Max Percent Error"])

# start the curve fitting process
expected_outputs = [evaluate_math_expression(
    expression, x) for x in input_values]

# Step: Construct a polynomial curve fitting function
degree = 3  # Set the degree of the polynomial here
coefficients = np.polyfit(input_values, expected_outputs, degree)
polynomial = np.poly1d(coefficients)

# Step 7: Evaluate the accuracy of the curve fitting function
max_percent_error_CF = 0
for input_value, expected_output in zip(input_values, expected_outputs):
    # Evaluate the curve fitting function with the input value and return the result
    output = polynomial(input_value)

    # Calculate the percent error
    percent_error = abs((output - expected_output) / expected_output) * 100

    # Update the maximum percent error if needed
    max_percent_error_CF = max(max_percent_error_CF, percent_error)

# Print the maximum percent error for the curve fitting function
print(f"Curve Fitting Function:\n{polynomial}")
print(f"Curve Fitting Function Max Percent Error: {max_percent_error_CF}%")


# Step 8: Evaluate generated functions for scalability
function_stats = []

for i, function in enumerate(valid_generated_functions, start=1):
    c_file_path = f"function{i}.c"
    bc_file_path = f"function{i}.bc"
    klee_out_dir = f"klee-out-{i - 1}"

    print(f"Testing Function {i} with Klee:\n")

    # Create an empty file using the `touch` command
    subprocess.run(["touch", c_file_path])

    # Write the complete code with main function
    complete_code = f"""
    #include <klee/klee.h>
    {function}

    int main(void) {{
        int x;
        klee_make_symbolic(&x, sizeof(x), "x");
        return compute_result(x);
    }}
"""

    with open(c_file_path, "w") as file:
        file.write(complete_code)

   # Compile the C file with Clang to generate LLVM bitcode
    subprocess.run(["clang", "-I", "../../include", "-emit-llvm",
                   "-c", "-g", "-O0", c_file_path, "-o", bc_file_path])

    # Run Klee on the compiled file
    subprocess.run(["klee", bc_file_path])

    # Run klee-stats on Klee output directory
    klee_stats_output = subprocess.run(
        ["klee-stats"] + ["--print-all"] + [klee_out_dir], capture_output=True, text=True).stdout

    # Parse Klee stats
    stats = parse_klee_stats(klee_stats_output)
    function_stats.append({
        "function": function,
        "stats": stats,
    })

    # Clean up the generated files
    subprocess.run(["rm", c_file_path, bc_file_path])
    # subprocess.run(["rm", "-r", klee_out_dir])

function_stats_scalability = []
# Print the statistics for each function
for i, function_stat in enumerate(function_stats, start=1):
    function = function_stat["function"]
    stats = function_stat["stats"]

    instruction_count = stats.get('Instrs')
    execution_time = stats.get('Time(s)')
    state_count = stats.get('States')
    memory_count = stats.get('Mem(MiB)')

    function_stats_scalability.append({
        "function": function,
        "stats": {
            "state": state_count,
        }
    })
    # Add more variables for other data points of interest

    print(f"Function {i}:")
    print(function)
    print(f"Instruction Count: {instruction_count}")
    print(f"Execution Time: {execution_time}")
    print(f"State Count: {state_count}")
    print(f"Memory Count: {memory_count}")
    # Print more statistics as desired
    print("\n")


# Sort the function stats based on the scalability score
function_stats_scalability = sorted(
    function_stats_scalability, key=lambda x: x["stats"]["state"])


def clear_klee_output_directories():
    directory = "."  # Path to the directory where the KLEE output directories are located

    # Get a list of all directories in the specified directory
    directories = [name for name in os.listdir(
        directory) if os.path.isdir(os.path.join(directory, name))]

    # Filter and delete directories starting with "klee-out-" and "klee-last"
    for dir_name in directories:
        if dir_name.startswith("klee-out-") or dir_name == "klee-last":
            dir_path = os.path.join(directory, dir_name)
            try:
                subprocess.run(["rm", "-r", dir_path])  # Remove the directory
            except OSError as e:
                print(f"Error deleting directory: {e}")

# Print the top 3 functions based on accuracy
# print("************************")
# print("TOP 3 FOR ACCURACY")
# for i, function_data in enumerate(function_stats_accuracy[:3], start=1):
#     function = function_data["function"]
#     max_percent_error = function_data["stats"]["Max Percent Error"]
#     print(f"Function {i}:")
#     print(function)
#     print(f"Max Percent Error: {max_percent_error}%")
#     print()


# Print the top 3 functions based on scalability
print("************************")
print("TOP 3 FOR SCALABILITY")
for i, function_data in enumerate(function_stats_scalability[:3], start=1):
    function = function_data["function"]
    state = function_data["stats"]["state"]
    print(f"Function {i}:")
    print(function)
    print(f"Number of states: {state}")
    print()

# Check if accuracy needs to be improved
max_percent_error_threshold = 30.0  # Set your desired threshold here
if function_stats_accuracy[0]["stats"]["Max Percent Error"] > max_percent_error_threshold:
    # Generate a new prompt to request more accurate functions
    new_prompt = f"The most accurate generated function has a maximum percent error of {function_stats_accuracy[0]['stats']['Max Percent Error']}%, which is above our desired threshold. Please generate more accurate integer-based functions in C with the name compute_result that compute the result of the expression '{expression}' without using floating-point operations (like sqrt) or iterations (like while and for loop). The functions should take the same parameters but in integer-based form and return an integer result and have a maximum percent error below {max_percent_error_threshold}%. Please ouput exact like this format:\n\nint compute_result(int x) {{\n *your code goes here*\n}} (remember the open bracket is at same line as function name do not break line for open bracket."
    new_response_choices = chat_with_gpt(new_prompt)
    new_generated_functions = [choice.text.strip()
                               for choice in new_response_choices]
    # Remove duplicate functions
    new_generated_functions = list(set(new_generated_functions))
    # Add new generated functions to existing list
    generated_functions.extend(new_generated_functions)

    # Re-evaluate all generated functions and get updated stats
    valid_generated_functions, function_stats_accuracy = evaluate_functions(
        generated_functions, expression, input_values)
    # Re-sort based on maximum percent error
    function_stats_accuracy = sorted(
        function_stats_accuracy, key=lambda x: x["stats"]["Max Percent Error"])

    # Print the updated top 3 most accurate functions
    top_3_functions = []
    print("UPDATED TOP 3 FOR ACCURACY (INITIAL)")
    for i in range(3):
        function_stat = function_stats_accuracy[i]
        function = function_stat["function"]
        max_percent_error = function_stat["stats"]["Max Percent Error"]
        print(f"Function {i + 1}:")
        print(function)
        print(f"Max Percent Error: {max_percent_error}%")
        print()
        top_3_functions.append(function)

# Generate input values for final evaluation
start = 0
end = 10000
input_values_top3 = list(range(start, end + 1))

# Evaluate top 3 functions with larger range of input values
valid_top_3_functions, top_3_function_stats_accuracy = evaluate_functions(
    top_3_functions, expression, input_values_top3)

# Sort the top 3 function stats based on maximum percent error (accuracy)
top_3_function_stats_accuracy = sorted(
    top_3_function_stats_accuracy, key=lambda x: x["stats"]["Max Percent Error"])

# Print the top 3 most accurate functions based on final evaluation
print("************************")
print("TOP 3 FOR ACCURACY (FINAL EVALUATION)")
for i, function_data in enumerate(top_3_function_stats_accuracy[:3], start=1):
    function = function_data["function"]
    max_percent_error = function_data["stats"]["Max Percent Error"]
    print(f"Function {i}:")
    print(function)
    print(f"Max Percent Error: {max_percent_error}%")
    print()

# Call the function to clear the KLEE output directories
clear_klee_output_directories()
