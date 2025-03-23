# %%
import os  # noqa
import shutil  # noqa
import urllib  # noqa
from pathlib import Path  # noqa

# url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
# urllib.request.urlretrieve(url, file_path)

external_data_path = Path(
    "/home/ds13/.bookmarks/shared-project-assets/Programming__/GitHub_datasets"
)
filename = "ghtorrent-2019-01-07_dataset"

data_path = Path("data")

data_path.mkdir(exist_ok=True)

raw_data_path = data_path / "raw"
raw_data_path.mkdir(exist_ok=True)

dataset_path = raw_data_path / filename

if not dataset_path.exists():
    shutil.unpack_archive(
        external_data_path / (filename + ".zip"), extract_dir=raw_data_path
    )
    # file_path.unlink()  # Remove archive after extracting it.

# %%
import pandas as pd  # noqa

CHUNKSIZE = 100_000

dataset_file_path = dataset_path / "ghtorrent-2019-01-07.csv"
data = pd.read_csv(dataset_file_path, chunksize=CHUNKSIZE, on_bad_lines="warn", sep=",")
data_part1 = pd.read_csv(dataset_file_path, nrows=CHUNKSIZE)

# %%
from IPython.display import display  # noqa

# df = data
# with df as reader:
# data_chunk = reader.get_chunk(CHUNKSIZE)
# display(data_chunk.head())


# %%
data_part1.head()

# %%
data_part1.shape

# %%
data_part1.dtypes

# %% [markdown]
# Некоторые колонки имеют неточные типы данных, их следует преобразовать

# %%
data_part1["commit_date"] = pd.to_datetime(data_part1["commit_date"], utc=True)
data_part1 = data_part1.astype(
    {
        "actor_id": "uint64",
        "comment_id": "uint64",
        "author_id": "uint64",
        "pr_id": "uint64",
        "c_id": "uint64",
    }
)
data_part1.dtypes, data_part1["commit_date"]

# %% [markdown]
# Посмотрим, как можно оптимизировать наш большой набор данных.

# %%
data_part1.memory_usage(deep=True)

# %% [markdown]
# Как видно, наибольший размер занимают строки. Заметим, что исходя из знаний о предметной области,
# многие из них будут часто повторяться. Комментарии, как правило отличны, в то время как
# пользователи и репозитории в которых они оставляют комментарии, будут повторяться.

# Языки и вовсе являются ограниченным множеством.

# Трансформируем соответствующие колонки в категориальный тип.

# %%
data_part1 = data_part1.astype(
    {
        "actor_login": "category",
        "language": "category",
        "repo": "category",
        "author_login": "category",
    }
)
data_part1.dtypes, data_part1.memory_usage(deep=True)

# %% [markdown]
# Как видно, наша гипотеза оказалась верной, все колонки кроме 'actor_login'
# стали занимать меньше места в памяти. Пока что 'actor_login' оставим
# категориальным, так как разница в памяти относительно невелика.

# %%
# Проверим наличие пустых значений
# Цикл по колонкам датасета
columns_number_of_empty_columns: dict[str, int] = {}

chunks1 = data
with data as reader:
    data_chunk = reader.get_chunk(CHUNKSIZE)

    for col in data_chunk.columns:
        # Количество пустых значений - все значения заполнены
        temp_null_count = data_chunk[data_chunk[col].isnull()].shape[0]
        columns_number_of_empty_columns[col] = (
            columns_number_of_empty_columns.get(col, 0) + temp_null_count
        )


pd.DataFrame(
    data={
        "Колонка": columns_number_of_empty_columns.keys(),
        "Кол-во пустых значений": columns_number_of_empty_columns.values(),
    }
)

# %% [markdown]
# > Далее набор данных `data` будет выкидывать ошибку. Это связано с тем, что
# итератор закончил свою работу. Нужно заново читать набор данных.
