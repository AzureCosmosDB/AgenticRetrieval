Run script without using graph augmentation (references):

```
python answer_questions_ref.py --config config.pubmed.yaml --questions ../data/complex_questions.json --output output/answers/answers_top15.json --top 15 --ref-k 0
```

This script will produce an answer file where each answer is generated from 15 documents.

Run script with graph augmentation (references):

```
python answer_questions_ref.py --config config.pubmed.yaml --questions ../data/complex_questions.json --output output/answers_ref/answers_top5_refk4.json --top 5 --ref-k 4
```

The last script will produce an answer file where each answer is generated from 16.78 documents on average (the average over 100 questions).

Now we can compare the 2 produced answer files using ragas:

```
python evaluate_answers_ragas.py --base output/answers/answers_top15.json --other output/answers_ref/answers_top5_refk4.json --base-name vector15 --other-name refs_5_4 --output output/eval_ragas.json
```

This should produce comparison similar to this:


<pre>
=====================================================================
RAGAS Comparison: vector15 vs refs_5_4
======================================================================
Metric                           vector15        refs_5_4      delta
----------------------------------------------------------------------
comprehensiveness                  0.9278          0.9240   -0.0038
evidence_use                       0.8940          0.9060 +   0.0120
accuracy                           0.8908          0.8899   -0.0009
clarity                            0.9329          0.9349 +   0.0020
relevance                          0.9429          0.9414   -0.0015
======================================================================
</pre>