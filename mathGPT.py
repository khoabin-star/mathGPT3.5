import openai
import ast
import math
import re
import subprocess
import os
import numpy as np
from math import gcd
from functools import reduce
from fractions import Fraction
from decimal import Decimal, getcontext

# Set up your OpenAI API credentials
openai.api_key = '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'

# Define the function to interact with ChatGPT
def chat_with_gpt(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        temperature=0.4,
        n=5,
        stop=None,
    )
    return response.choices


def evaluate_functions(generated_functions, expression, input_values):
    function_stats_accuracy = []
    valid_generated_functions = []
    for i, function in enumerate(generated_functions, start=1):
        max_percent_error = 0
        absolute_error = 0
        sum_absolute_error = 0
        max_absolute_error = 0
        avg_absolute_error = 0

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
                if function.count('{') != function.count('}'):
                    raise SyntaxError("Mismatched brackets")
                ast.parse(python_code)
                if function not in valid_generated_functions:
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

            for x, y in input_values:
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
                    
                    # Calculate the absolute error
                    absolute_error = abs(output - expected_output)

                    # Update the sum of absolute errors
                    sum_absolute_error += absolute_error

                    # Update the maximum absolute error if needed
                    max_absolute_error = max(max_absolute_error, absolute_error)

                    # # Print intermediate values for debugging
                    # print(f"Function: {function}")
                    # print(f"Input value: {input_value}")
                    # print(f"Output: {output}")
                    # print(f"Expected output: {expected_output}")
                    # print(f"Percent error: {percent_error}%")
                    # print(f"Absolute error: {absolute_error}")
                    # print()


                except NameError as e:
                    if "name 'result' is not defined" in str(e):
                        print(f"Error: Generated function is invalid: {e}")
                        valid_generated_functions.remove(function)
                        break
                    else:
                        print(f"Error evaluating function with input {input_value}: {e}")
                        break
                except Exception as e:
                    print(f"Error evaluating function with input {input_value}: {e}")
                    break

            else:
                # Calculate the average absolute error
                avg_absolute_error = sum_absolute_error / len(input_values)

                 # Check if a function with the same accuracy stats already exists in the list
                existing_function_data = next((data for data in function_stats_accuracy if data["stats"]["Max Percent Error"] == max_percent_error and data["stats"]["Max Absolute Error"] == max_absolute_error and data["stats"]["Avg Absolute Error"] == avg_absolute_error), None)

                if existing_function_data is None:
                    # Add the function and its stats to the list
                    function_stats_accuracy.append({
                        "function": function,
                        "stats": {
                            "Max Percent Error": max_percent_error,
                            "Max Absolute Error": max_absolute_error,
                            "Avg Absolute Error": avg_absolute_error,
                        }
                    })

        except Exception as e:
            print(f"Error compiling function: {e}")
    return valid_generated_functions, function_stats_accuracy


# Implement the evaluate_math_expression function
def evaluate_math_expression(expression, input_value):
    # Create a dictionary containing all the functions and constants from the math module
    math_namespace = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}

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

    while any(keyword in function for keyword in disallowed_keywords) or has_floating_point_value:
        # Generate a new prompt to request a corrected function
        new_prompt = f"The following generated function contains disallowed keywords: {function}. Please generate a corrected version of this function that does not use external function (like sqrt), do not use floating-point values, and do not use iterations (like while and for loop). The function should take the same parameters but in integer-based form and return an integer result and have the highest approximation result compared to the original expression and please ouput exact like this format:\n\nint compute_result(int x) {{\n *your code goes here*\n}} (remember the open bracket is at same line as function name do not break line for open bracket."

        new_response_choices = chat_with_gpt(new_prompt)
        new_function = new_response_choices[0].text.strip()
        generated_functions[i] = new_function

        # Check if the new function contains disallowed keywords or floating-point values
        has_floating_point_value = re.search(r"\d+\.\d+", new_function) is not None
        if not any(keyword in new_function for keyword in disallowed_keywords) and not has_floating_point_value:
            break

# Step 5: Generate input values
# Example input values, adjust as needed
start = 0
end = 10
input_values = list(range(start, end + 1))

# Step 6: Evaluate generated functions and get function stats
valid_generated_functions, function_stats_accuracy = evaluate_functions(generated_functions, expression, input_values)

# Convert to dictionary to remove duplicates based on the function string
function_stats_dict = {f["function"]: f for f in function_stats_accuracy}

# Convert back to list
function_stats_accuracy = list(function_stats_dict.values())

# Sort the function stats based on the maximum percent error (accuracy)
function_stats_accuracy = sorted(function_stats_accuracy, key=lambda x: x["stats"]["Max Percent Error"])

# for i, function_data in enumerate(function_stats_accuracy, start=1):
#     print("*******Already supposed to be sorted\n")
#     funtion = function_data["function"]
#     max_percent_error = function_data["stats"]["Max Percent Error"]
#     max_absolute_error = function_data["stats"]["Max Absolute Error"]
#     average_absolute_error = function_data["stats"]["Avg Absolute Error"]
#     print(f"Generated function {i}:\n{function}")
#     print(f"Max percent error: {max_percent_error}%")
#     print(f"Max Absolute Error: {max_absolute_error}")
#     print(f"Average Absolute Error: {average_absolute_error}")
#     print()


def evaluate_scalability(valid_generated_functions):
    function_stats = []

    for i, function in enumerate(valid_generated_functions, start=1):
        c_file_path = f"function{i}.c"
        bc_file_path = f"function{i}.bc"
        klee_out_dir = f"klee-out-{i - 1}"

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
        subprocess.run(["clang", "-I", "../../include", "-emit-llvm", "-c", "-g", "-O0", c_file_path, "-o", bc_file_path])

        # Run Klee on the compiled file
        subprocess.run(["klee", bc_file_path])
        print(function)

        # Run klee-stats on Klee output directory
        klee_stats_output = subprocess.run(["klee-stats"] + ["--print-all"] + [klee_out_dir], capture_output=True, text=True).stdout

        # Parse Klee stats
        stats = parse_klee_stats(klee_stats_output)
        function_stats.append({
            "function": function, 
            "stats": stats,
        })

        # Clean up the generated files
        subprocess.run(["rm", c_file_path, bc_file_path])

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
    
    return function_stats_scalability

def clear_klee_output_directories():
    directory = "."  # Path to the directory where the KLEE output directories are located

    # Get a list of all directories in the specified directory
    directories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

    # Filter and delete directories starting with "klee-out-" and "klee-last"
    for dir_name in directories:
        if dir_name.startswith("klee-out-") or dir_name == "klee-last":
            dir_path = os.path.join(directory, dir_name)
            try:
                subprocess.run(["rm", "-r", dir_path])  # Remove the directory
            except OSError as e:
                print(f"Error deleting directory: {e}")


max_regenerations = 20 # Set the maximum number of times to regenerate functions here
regeneration_count = 0
top_3_functions = []
while regeneration_count < max_regenerations:
    # Generate a new prompt to request more accurate functions
    new_prompt = f"Please generate functions in C with the name compute_result that compute the result of the expression '{expression}' without using floating-point operations (like sqrt) or iterations (like while and for loop). The functions should take the same parameters but in integer-based form and return an integer result. The function does not need to be exact, but should provide a reasonable approximation of the '{expression}'. Please ouput exact like this format:\n\nint compute_result(int x) {{\n *your code goes here*\n}} (remember the open bracket is at same line as function name do not break line for open bracket."
    new_response_choices = chat_with_gpt(new_prompt)
    new_generated_functions = [choice.text.strip() for choice in new_response_choices]
    # Remove duplicate functions
    new_generated_functions = list(set(new_generated_functions))
    # Add new generated functions to existing list
    generated_functions.extend(new_generated_functions)

    # Re-evaluate all generated functions and get updated stats
    valid_generated_functions, function_stats_accuracy = evaluate_functions(generated_functions, expression, input_values)

    # Re-sort based on maximum percent error
    function_stats_accuracy = sorted(function_stats_accuracy, key=lambda x: x["stats"]["Max Percent Error"])

    # Check if the length of function_stats_accuracy is less than 3
    if len(function_stats_accuracy) < 3:
        # Append only the available functions to the top_3_functions list
        for function_stat in function_stats_accuracy:
            function = function_stat["function"]
            max_percent_error = function_stat["stats"]["Max Percent Error"]
            top_3_functions.append(function)
    else:
        # Append the top 3 functions to the top_3_functions list
        for i in range(3):
            function_stat = function_stats_accuracy[i]
            function = function_stat["function"]
            max_percent_error = function_stat["stats"]["Max Percent Error"]
            top_3_functions.append(function)

    regeneration_count += 1


# Generate input values for final evaluation
start = 0
end = 100
input_values_top3 = list(range(start, end + 1))

# Evaluate top 3 functions with larger range of input values
valid_top_3_functions, top_3_function_stats_accuracy = evaluate_functions(top_3_functions, expression, input_values_top3)

# Sort the top 3 function stats based on maximum percent error (accuracy)
top_3_function_stats_accuracy = sorted(top_3_function_stats_accuracy, key=lambda x: x["stats"]["Max Percent Error"])

# Print the top 3 most accurate functions based on final evaluation
print("************************")
print("TOP 3 FOR ACCURACY (FINAL EVALUATION)")
if len(top_3_function_stats_accuracy) < 3:
    print("There are no top 3\n")
    for i, function_data in enumerate(top_3_function_stats_accuracy, start=1):
        function = function_data["function"]
        max_percent_error = function_data["stats"]["Max Percent Error"]
        max_absolute_error = function_data["stats"]["Max Absolute Error"]
        average_absolute_error = function_data["stats"]["Avg Absolute Error"]

        print(f"Generated function {i}:\n{function}")
        print(f"Max percent error: {max_percent_error}%")
        print(f"Max Absolute Error: {max_absolute_error}")
        print(f"Average Absolute Error: {average_absolute_error}")
        print()
else:
    for i, function_data in enumerate(top_3_function_stats_accuracy[:3], start=1):
        function = function_data["function"]
        max_percent_error = function_data["stats"]["Max Percent Error"]
        max_absolute_error = function_data["stats"]["Max Absolute Error"]
        average_absolute_error = function_data["stats"]["Avg Absolute Error"]
        print(f"Function {i}:")
        print(function)
        print(f"Max Percent Error: {max_percent_error}%")
        print(f"Max Absolute Error: {max_absolute_error}")
        print(f"Average Absolute Error: {average_absolute_error}")
        print()

def curve_fitting_to_integer(expression, input_values, degree):
    # Set the precision of the Decimal class
    getcontext().prec = 28

    # Start the curve fitting process
    expected_outputs = [evaluate_math_expression(expression, x) for x in input_values]

    # Construct a polynomial curve fitting function
    coefficients = np.polyfit(input_values, expected_outputs, degree)
    polynomial = np.poly1d(coefficients)
    print(f"Polynomial: {polynomial}")

    # Check if all coefficients are integers
    if all(coeff.is_integer() for coeff in coefficients):
        print("****THIS IS INTEGER****")
        return polynomial, 1

    # Convert coefficients to Decimals
    decimals = [Decimal(coeff) for coeff in coefficients]

    # Convert Decimals to fractions
    fractions = [Fraction(decimal) for decimal in decimals]

    # Find LCM of denominators
    denominators = [fraction.denominator for fraction in fractions]
    lcm_denominator = reduce(lambda a, b: a * b // gcd(a, b), denominators)

    # Convert coefficients to integers
    integer_coefficients = [int(fraction * lcm_denominator) for fraction in fractions]
    
    # Construct integer polynomial
    integer_polynomial = np.poly1d(integer_coefficients)

    return integer_polynomial, lcm_denominator

def generate_c_function(expression, input_values, degree):
    # Use the curve_fitting_to_integer function to construct an integer polynomial
    integer_polynomial, lcm_denominator = curve_fitting_to_integer(expression, input_values, degree)

    # Get the coefficients of the polynomial
    coefficients = integer_polynomial.coefficients

    # Calculate the GCD of all the coefficients
    gcd_coefficients = reduce(gcd, map(abs, coefficients))

    # Divide all the coefficients by their GCD
    reduced_coefficients = [coeff // gcd_coefficients for coeff in coefficients]

    # Construct the C-style function
    c_function = "int compute_result(int x) {\n"
    c_function += "    int result = 0;\n"
    for i, coeff in enumerate(reduced_coefficients):
        power = len(reduced_coefficients) - i - 1
        if power == 0:
            c_function += f"    result += {coeff};\n"
        elif power == 1:
            c_function += f"    result += {coeff} * x;\n"
        else:
            c_function += f"    result += {coeff} * "
            for j in range(power - 1):
                c_function += "x * "
            c_function += "x;\n"
    c_function += f"    return result / {lcm_denominator // gcd_coefficients};\n"
    c_function += "}"

    return c_function

degree = 1
# Generate the C-style function using the generate_c_function function
c_function = generate_c_function(expression, input_values, degree)

# Evaluate the generated function using the evaluate_functions function
valid_generated_functions, curve_fit_function_stats_accuracy = evaluate_functions([c_function], expression, input_values_top3)

# Print the accuracy statistics for the generated function
curve_fit_function_stat = curve_fit_function_stats_accuracy[0]
curve_fit_function = curve_fit_function_stat["function"]
curve_fit_max_percent_error = curve_fit_function_stat["stats"]["Max Percent Error"]
curve_fit_max_absolute_error = curve_fit_function_stat["stats"]["Max Absolute Error"]
curve_fit_average_absolute_error = curve_fit_function_stat["stats"]["Avg Absolute Error"]
print(f"Generated Function:")
print(curve_fit_function)
print(f"Max Percent Error: {curve_fit_max_percent_error}%")
print(f"Max Absolute Error: {curve_fit_max_absolute_error}")
print(f"Average Absolute Error: {curve_fit_average_absolute_error}")

# Define a list containing the top 3 functions for accuracy (final evaluation) and the curve fitting function
if len(function_stats_accuracy) < 3:
    print(f"There are only {len(function_stats_accuracy)} functions in the top 3.")
else:
    functions = [top_3_function_stats_accuracy[0]["function"], top_3_function_stats_accuracy[1]["function"], top_3_function_stats_accuracy[2]["function"], curve_fit_function]
    # Evaluate the scalability of these functions using the evaluate_scalability function
    function_stats_scalability = evaluate_scalability(functions)

    # Sort the function stats based on the desired scalability metric
    function_stats_scalability = sorted(function_stats_scalability, key=lambda x: x["stats"]["state"])

    # Print the scalability statistics for each function
    for i, function_data in enumerate(function_stats_scalability, start=1):
        function = function_data["function"]
        state = function_data["stats"]["state"]
        print(f"Function {i}:")
        print(function)
        print(f"Number of states: {state}")
        print()

# Call the function to clear the KLEE output directories
clear_klee_output_directories()

