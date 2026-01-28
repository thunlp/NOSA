from typing import List, Tuple

def sum_product(numbers: List[int]) -> Tuple[int, int]:
    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
    Empty sum should be equal to 0 and empty product should be equal to 1.
    >>> sum_product([])\n    (0, 1)\n    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """
    if not numbers:
        return 0, 1
    else:
        return sum(numbers), functools.reduce(lambda x, y: x * y, numbers)
```\n\nIn this function, we first check if the list is empty. If it is, we return a tuple with 0 and 1. If the list is not empty, we use the built-in `sum` function to calculate the sum of all the numbers in the list. We then use the `functools.reduce` function to calculate the product of all the numbers in the list. The `reduce` function applies a binary function (in this case, multiplication) to all elements of the list, reducing the list to a single value.\n\nThe `functools.reduce` function is a higher-order function that takes a function and a list as arguments, and applies the function to the first two elements of the list, then to the result and the next element, and so on, until it has processed the entire list. The `lambda` function is a small anonymous function that takes two arguments and returns their product.\n\nThis function will work for any list of integers, and will return a tuple with the sum and product of the numbers in the list.


from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n   numbers.sort()\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True\n    return False