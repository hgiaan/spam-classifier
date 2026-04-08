# Spam Detector - Naive Bayes from Scratch

A spam classifier built **entirely from scratch** in Python.

## Motivation
I wanted to understand how spam filters work. So I built one from scratch using only Python's standard library, implementing every step manually: probabilility estimation, text preprocessing.

## How it works

- Based on Naive Bayes - a probabilistic model based on Bayes' Theorem.
Given a message, it computes:
```text
score(SPAM) = log P(SPAM) + Σ log P(word | SPAM)
score(HAM)  = log P(HAM)  + Σ log P(word | HAM)
```
The label with the higher score wins.

- Key implementation decisions:
    + **Laplace Smoothing** — prevents zero-probability for unseen words
    + **Log-space computation** — avoids floating point underflow
    + **Gibberish filter** — handles messages with entirely unknown vocabulary
--> Full mathematical explanation: [docs/how_it_works.pdf]

## Results:
Base on the test, we have the result
```text
--- EVALUATION ---
Test set size : 1115 messages
Correct       : 1097
Accuracy      : 98.4%
```

Sample predictions:
> "Hey, are you free this afternoon?" → HAM  
> "WINNER!! Claim your £900 prize now!" → SPAM  
> "ofjoaiwjer" → SPAM (unknown vocabulary)

## What I learned
- How Bayes' Theorem applies to real classification problems
- Why floating-point underflow is a practical issue in ML and how to fix it
- The trade-off between model simplicity (Naive assumption) and real-world accuracy

## Limitations and Future Improvements:
- It reads "word by word", ignore word order (misses the context)
- It treats all words equally (doesn't help identify spam if words doesn't classify it's priority (red flags words: winner, urgent))
- In real life, the words “win” and “winning” belong to the same word family, so they are not counted as two separate words.
- Maybe build a web interface for everyone to use.

## Quick Start
```bash 
git clone https://github.com/hgiaan/spam-classifier
cd spam-classifier
python src/main.py
