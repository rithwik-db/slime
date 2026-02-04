# Copied from - https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/math_grader.py

import re

from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify


def is_equiv(gt_answer, predicted_answer):
    """Check if the predicted answer is equivalent to the ground truth answer."""
    if predicted_answer is None:
        return False

    if "boxed" in gt_answer:
        # Need to extract the answer from the boxed expression
        gt_answer = extract_answer(gt_answer)

    if "boxed" in predicted_answer:
        # Need to extract the answer from the boxed expression
        predicted_answer = extract_answer(predicted_answer)

    gt_answer = str(gt_answer)
    predicted_answer = str(predicted_answer)

    # Try to compare as MCQ options
    mcq_options = "ABCDEFGHIJ"
    norm_gt_mcq = gt_answer.strip()

    is_mcq = re.fullmatch("|".join(mcq_options), norm_gt_mcq)
    parsed_gt = parse(gt_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    parsed_pred = parse(predicted_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    if is_mcq and verify(parsed_gt, parsed_pred):
        return verify(parsed_gt, parsed_pred)

    # Additional normalization step
    gt_answer = _additional_normalization(gt_answer)
    predicted_answer = _additional_normalization(predicted_answer)

    # Try literal comparison
    literal_pattern = r"[a-zA-Z ,]+|[0-9 ]+"
    normalized_gt = normalize_latex(gt_answer, NormalizationConfig)
    normalized_pred = normalize_latex(predicted_answer, NormalizationConfig)
    is_literal = re.fullmatch(literal_pattern, normalized_gt) and re.fullmatch(literal_pattern, normalized_pred)
    is_normalized_equal = normalized_gt.replace(" ", "") == normalized_pred.replace(" ", "")

    if is_literal or is_normalized_equal:
        return is_normalized_equal

    # Fallback to symbolic comparison
    current_gt_answer = gt_answer
    current_predicted_answer = predicted_answer

    # math_verify.parse expects input to be in latex environment, e.g. $...$
    latex_env_search_pattern = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    if not re.search(latex_env_search_pattern, current_gt_answer, re.DOTALL):
        current_gt_answer = f"${current_gt_answer}$"
    if not re.search(latex_env_search_pattern, current_predicted_answer, re.DOTALL):
        current_predicted_answer = f"${current_predicted_answer}$"

    parsed_gt = parse(current_gt_answer, [LatexExtractionConfig()])
    parsed_pred = parse(current_predicted_answer, [LatexExtractionConfig()])

    return verify(parsed_gt, parsed_pred)


def _additional_normalization(expr):
    """Additional normalization of the expression."""
    # Remove % and \\% from the number
    percentage_pattern = r"^(\d+\.?\d*)(?:\\%|%)$"
    match_gt = re.fullmatch(percentage_pattern, expr)
    if match_gt:
        expr = match_gt.group(1)
    # Remove . corresponding to the end of sentence
    return expr.rstrip(".\\")


def extract_answer(
    string: str, extract_from_boxed: bool = True, extract_regex: str = r"The final answer is (.+)$", relaxed=False
):
    """Extract Answer String from \\boxed expression or based on regex
    If relaxed=True: try both methods, boxed first.
    If relaxed=False: use only one method based on extract_from_boxed flag.
    """
    if relaxed:
        return _search_boxed(string) or _search_regex(string, extract_regex)

    if extract_from_boxed:
        return _search_boxed(string)
    return _search_regex(string, extract_regex)


def _search_regex(string: str, regex: str):
    """Search for the regex in the string."""
    match = re.findall(regex, string)
    if match:
        return match[-1]
    return None


def _search_boxed(string: str):
    """Search for the boxed expression in the string."""
    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None


def get_math_verifier_reward(response, label, verifier_score=4.0, format_score=1.0):
    """
    Reward function for verifying math answers and checking boxed formatting.
    
    Args:
        response: The model's response string
        label: The ground truth answer
        verifier_score: Reward value for correct math verification (default: 4.0)
        format_score: Reward value for correct format (default: 1.0)
    
    Returns:
        float: Total reward:
               - 0.0 if no boxed answer found
               - format_score (1.0) if boxed answer exists but incorrect
               - format_score + verifier_score (5.0) if boxed answer correct
    """
    # Extract answer from response (expects boxed format like \boxed{...})
    pred_answer = extract_answer(response)
    
    # No valid boxed answer found
    if pred_answer is None:
        return 0.0
    
    # Format is correct (boxed answer exists)
    total_reward = format_score
    
    # Check if label is valid
    if label == "" or label is None:
        return total_reward  # Only format reward, no ground truth to verify
    
    # Convert label to string for comparison
    gt_answer = str(label)
    
    # Check if answer is equivalent to ground truth
    if is_equiv(gt_answer, pred_answer):
        total_reward += verifier_score
    
    return total_reward