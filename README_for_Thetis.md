# Instructions
This repo is a fork of the [TURL](https://github.com/sunlab-osu/TURL) repo with the goal of generating table embeddings that are used in the experiments of the [Semantic Table Search Dataset](https://github.com/EDAO-Project/SemanticTableSearchDataset).


Begin by setting up a virtual environment
```
python3.8 -m venv env
source env/bin/activate
```

Follow the instructions in the original [README.md](README.md) to install all dependences and models required by TURL.

## Extract TURL per table embeddings
Before running TURL over each table corpus populate the `input/wikipages_2013/` and `input/wikipages_2019/` directories with the table files in .csv format.

TURL table embeddings for each table in the Wikitables 2013 corpus:
```
python TURL_get_embeddings.py --input_dir input/wikipages_2013/ --output_dir output/embeddings/wikipages_2013/
```

TURL table embeddings for each table in the Wikitables 2019 corpus:
```
python TURL_get_embeddings.py --input_dir input/wikipages_2019/ --output_dir output/embeddings/wikipages_2019/
```

Similarly generate the 'table' embeddings for each query in the 2013 and 2019 corpora respectively:

```
python TURL_get_embeddings.py --input_dir input/wikipages_2013_queries/ --output_dir output/embeddings/wikipages_2013_queries/
```

```
python TURL_get_embeddings.py --input_dir input/wikipages_2019_queries/ --output_dir output/embeddings/wikipages_2019_queries/
```