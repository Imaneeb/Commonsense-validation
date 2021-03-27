"""TODO(commonsense_qa): Add a description here."""

from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np 
import json
import requests
import io

import datasets


# TODO(commonsense_qa): BibTeX citation
_CITATION = """\
@InProceedings{commonsense_QA)

"""

# TODO(commonsense_qa):
_DESCRIPTION = """\
CommonsenseQA is a new multiple-choice question answering dataset 
"""

_URL = "https://raw.githubusercontent.com/Imanebouayad/commonsense_dataset/main/"
_URLS = {
    "train": _URL + "task2_train.csv",
    "dev": _URL + "task2_dev.csv",
    "test": _URL + "task2_test.csv",
}



class CommonsenseQa(datasets.GeneratorBasedBuilder):
    """TODO(commonsense_qa): Short description of my dataset."""

    # TODO(commonsense_qa): Set up version.
    VERSION = datasets.Version("1.0.0")

   
    def _info(self):
        # These are the features of your dataset like images, labels ...
        features = datasets.Features(
            {
                "answerKey": datasets.Value("string"),
                "statement": datasets.Value("string"),
                "choices": datasets.features.Sequence(
                    {
                        "label": datasets.Value("string"),
                        "text": datasets.Value("string"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=features,
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/Imanebouayad/commonsense_dataset",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        download_urls = _URLS

        downloaded_files = dl_manager.download_and_extract(download_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"],
                "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        # TODO(commonsense_qa): Yields (key, example) tuples from the dataset
        
        df= pd.read_csv(filepath,encoding="utf-8")
        df=df.drop(['id'], axis = 1)
        cols = list(df)
        df[cols] = df[cols].astype(str)
        for id_,row in df.iterrows():
            statement = row [0]
            text = [row[1],row[2],row[3]]
            labels = ['A', 'B', 'C']
            answerkey = row[-1]
            yield id_, {
                    "answerKey": answerkey,
                    "statement": statement,
                    "choices": {"label": labels, "text": text},
            }
