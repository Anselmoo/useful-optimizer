"""Benchmark functions for optimization problems."""

from __future__ import annotations

import numpy as np


# Non-Centered Ackley Function
def shifted_ackley(x: np.ndarray, shift: tuple = (1, 0.5)) -> float:
    """Shifted Ackley function.

    Args:
        x (np.ndarray): Input vector.
        shift (np.ndarray): Shift vector.

    Returns:
        float: Output value.
    """
    return ackley(np.array([x[i] - shift[i] for i in range(len(x))]))


def sphere(x: np.ndarray) -> float:
    """Sphere function.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Output value.
    """
    return np.sum(x**2)


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function.

    Args:
        x (np.ndarray): Input array of shape with larger than 2,
            representing the coordinates.

    Returns:
        float: Output value.
    """
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)


# Ackley Function
def ackley(x: np.ndarray) -> float:
    """Ackley function.

    Args:
        x (np.ndarray): Input array of shape (2,) representing the coordinates.

    Returns:
        float: Output value.
    """
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
        - np.exp(0.5 * (np.cos(2.0 * np.pi * x[0]) + np.cos(2.0 * np.pi * x[1])))
        + np.e
        + 20
    )


# Griewank Function
def griewank(x: np.ndarray) -> float:
    """Griewank function.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Output value.
    """
    return (
        1
        + (1 / 4000) * np.sum(x**2)
        - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1))))
    )


# Rastrigin Function
def rastrigin(x: np.ndarray) -> float:
    """Rastrigin Function.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: The value of the Rastrigin function at the given input.
    """
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


# Schwefel Function
def schwefel(x: np.ndarray) -> float:
    """Schwefel function.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Output value.
    """
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


# Levi Function
def levi(x: np.ndarray) -> float:
    """Levi function.

    Args:
        x (np.ndarray): Input array of shape (2,) representing the coordinates.

    Returns:
        float: The value of the Levi function at the given coordinates.
    """
    return (
        np.sin(3 * np.pi * x[0]) ** 2
        + (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2)
        + (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2)
    )


# Himmelblau Function
def himmelblau(x: np.ndarray) -> float:
    """Calculate the value of the Himmelblau function for the given input.

    Args:
        x (np.ndarray): The input array of shape (2,) representing the coordinates.

    Returns:
        float: The value of the Himmelblau function.
    """
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


# Eggholder Function
def eggholder(x: np.ndarray) -> float:
    """Calculate the value of the Eggholder function for the given input.

    Args:
        x (np.ndarray): The input array of shape (2,) representing the coordinates.

    Returns:
        float: The value of the Eggholder function.
    """
    return -(x[1] + 47) * np.sin(np.sqrt(abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(
        np.sqrt(abs(x[0] - (x[1] + 47)))
    )


# Beale Function
def beale(x: np.ndarray) -> float:
    """Calculate the value of the Beale function for the given input.

    Args:
        x (np.ndarray): The input array of shape (2,) representing the coordinates.

    Returns:
        float: The value of the Beale function.
    """
    return (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )


# Goldstein-Price Function
def goldstein_price(x: np.ndarray) -> float:
    """Calculate the value of the Goldstein-Price function for the given input.

    Args:
        x (np.ndarray): The input array of shape (2,) representing the coordinates.

    Returns:
        float: The value of the Goldstein-Price function.
    """
    return (
        1
        + (x[0] + x[1] + 1) ** 2
        * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
    ) * (
        30
        + (2 * x[0] - 3 * x[1]) ** 2
        * (
            18
            - 32 * x[0]
            + 12 * x[0] ** 2
            + 48 * x[1]
            - 36 * x[0] * x[1]
            + 27 * x[1] ** 2
        )
    )


# Booth Function
def booth(x: np.ndarray) -> float:
    """Calculate the value of the Booth function for the given input.

    Args:
        x (np.ndarray): The input array of shape (2,) representing the coordinates.

    Returns:
        float: The value of the Booth function.
    """
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


# Bukin Function
def bukin(x: np.ndarray) -> float:
    """Calculate the value of the Bukin function for the given input.

    Args:
        x (np.ndarray): The input array of shape (2,) representing the coordinates.

    Returns:
        float: The value of the Bukin function.
    """
    return 100 * np.sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10)


# Matyas Function
def matyas(x: np.ndarray) -> float:
    """Computes the Matyas function for the given input.

    Args:
        x (np.ndarray): The input array of shape (2,) representing the coordinates.

    Returns:
        float: The value of the Matyas function.
    """
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


# Levi Function N.13
def levi_n13(x: np.ndarray) -> float:
    """Calculate the value of the Levi N.13 function for the given input.

    Args:
        x (np.ndarray): The input array of shape (2,) containing the values of x[0]
            and x[1].

    Returns:
        float: The calculated value of the Levi N.13 function.
    """
    return (
        np.sin(3 * np.pi * x[0]) ** 2
        + (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2)
        + (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2)
    )


# Three-hump Camel Function
def three_hump_camel(x: np.ndarray) -> float:
    """Calculate the value of the Three-hump Camel function for the given input.

    Args:
        x (np.ndarray): Input vector of shape (2,).

    Returns:
        float: Output value of the Three-hump Camel function.
    """
    return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2


# Easom Function
def easom(x: np.ndarray) -> float:
    """Calculate the value of the Easom function for the given input.

    Args:
        x (np.ndarray): Input vector of shape (2,).

    Returns:
        float: Output value of the Easom function.
    """
    return (
        -np.cos(x[0])
        * np.cos(x[1])
        * np.exp(-((x[0] - np.pi) ** 2) - (x[1] - np.pi) ** 2)
    )


def cross_in_tray(x: np.ndarray) -> float:
    """Cross-in-Tray function.

    Args:
        x (np.ndarray): Input vector of shape (2,).

    Returns:
        float: Output value of the Cross-in-Tray function.
    """
    return (
        -0.0001
        * (
            abs(
                np.sin(x[0])
                * np.sin(x[1])
                * np.exp(abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
            )
            + 1
        )
        ** 0.1
    )


def hold_table(x: np.ndarray) -> float:
    """Hold Table function.

    Args:
        x (np.ndarray): Input vector of shape (2,).

    Returns:
        float: Output value of the Hold Table function.
    """
    return -np.abs(
        np.sin(x[0])
        * np.sin(x[1])
        * np.exp(abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
    )


def mccormick(x: np.ndarray) -> float:
    """McCormick function.

    Args:
        x (np.ndarray): Input vector of shape (2,).

    Returns:
        float: Output value of the McCormick function.
    """
    return np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1
