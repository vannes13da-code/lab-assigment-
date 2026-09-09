
# K-MEANS CLUSTERING
# SEARCH FOR THE OPTIMAL VALUE OF K

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

archivo = r"C:\Documentos\2SEM2026\intelligence Artificial\lab assigment 1\mdi_personasdesaparecidas_pm_2026_enero_julio.xlsx"

excel = pd.ExcelFile(archivo)


for i, hoja in enumerate(excel.sheet_names):
    print(i, "->", hoja)


sheet_name = None

for hoja in excel.sheet_names:

    if "pdesaparecidas" in hoja.lower():
        sheet_name = hoja
        break


# If it is not found, use the second sheet
if sheet_name is None:

    if len(excel.sheet_names) > 1:
        sheet_name = excel.sheet_names[1]
    else:
        sheet_name = excel.sheet_names[0]


print("\nSelected sheet:")
print(sheet_name)

data = pd.read_excel(
    archivo,
    sheet_name=sheet_name
)

print("\nFirst rows:")

print(data.head())


print("\nOriginal dimensions:")

print(data.shape)


print("\nOriginal columns:")

print(data.columns.tolist())


data = data.dropna(
    axis=1,
    how="all"
)



data = data.dropna(
    axis=0,
    how="all"
)
data = data.reset_index(drop=True)

unnamed_columns = [
    column
    for column in data.columns
    if str(column).lower().startswith("unnamed")
]

data = data.drop(
    columns=unnamed_columns,
    errors="ignore"
)
print(data.head())

print("\nDimensions:")

print(data.shape)


print("\nColumns:")

print(data.columns.tolist())

for column in data.columns:

    if data[column].dtype == "object":

        converted = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        original_non_null = data[column].notna().sum()

        converted_non_null = converted.notna().sum()

       

        if original_non_null > 0:

            ratio = (
                converted_non_null
                /
                original_non_null
            )

            if ratio >= 0.80:
                data[column] = converted

numeric_columns = data.select_dtypes(
    include=np.number
).columns.tolist()


categorical_columns = data.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print(
    "\nNumerical columns (" +
    str(len(numeric_columns)) +
    "):"
)

print(numeric_columns)


print(
    "\nCategorical columns (" +
    str(len(categorical_columns)) +
    "):"
)

print(categorical_columns)

high_cardinality_columns = []


for column in categorical_columns:

    unique_values = data[column].nunique(
        dropna=True
    )

    total_values = data[column].notna().sum()

    if total_values > 0:

        unique_ratio = (
            unique_values
            /
            total_values
        )

        if (
            unique_values > 30
            and unique_ratio > 0.70
        ):

            high_cardinality_columns.append(
                column
            )


if len(high_cardinality_columns) > 0:

    print(
        "\nHigh-cardinality columns removed:"
    )

    print(high_cardinality_columns)

    data_model = data.drop(
        columns=high_cardinality_columns
    )

else:

    data_model = data.copy()


numeric_columns = data_model.select_dtypes(
    include=np.number
).columns.tolist()


categorical_columns = data_model.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()


print("\nVariables used by the model:")

print(
    "\nNumerical:",
    numeric_columns
)

print(
    "\nCategorical:",
    categorical_columns
)



if (
    len(numeric_columns) == 0
    and len(categorical_columns) == 0
):

    raise ValueError(
        "No useful variables were found for clustering."
    )


numeric_pipeline = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",

            StandardScaler()
        )

    ]

)



categorical_pipeline = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )

    ]

)

transformers = []


if len(numeric_columns) > 0:

    transformers.append(

        (
            "numeric",

            numeric_pipeline,

            numeric_columns
        )

    )


if len(categorical_columns) > 0:

    transformers.append(

        (
            "categorical",

            categorical_pipeline,

            categorical_columns
        )

    )


preprocessor = ColumnTransformer(
    transformers=transformers
)

X_preprocessed = preprocessor.fit_transform(
    data_model
)


print(
    "\nOriginal shape:",
    data_model.shape
)


print(
    "Shape after preprocessing:",
    X_preprocessed.shape
)

number_of_samples = X_preprocessed.shape[0]


if number_of_samples < 3:

    raise ValueError(
        "The dataset has too few observations "
        "for K-Means."
    )

max_k = min(
    10,
    number_of_samples - 1
)


K_values = range(
    2,
    max_k + 1
)

inertias = []

silhouette_scores = []


print("\n============================================")
print("TESTING DIFFERENT VALUES OF K")
print("============================================")


for k in K_values:

    model = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10

    )


    labels = model.fit_predict(
        X_preprocessed
    )


    inertia = model.inertia_


    silhouette = silhouette_score(

        X_preprocessed,

        labels

    )


    inertias.append(
        inertia
    )


    silhouette_scores.append(
        silhouette
    )


    print(
        "k =",
        k,
        "| Inertia =",
        round(inertia, 2),
        "| Silhouette Score =",
        round(silhouette, 4)
    )


best_position = np.argmax(
    silhouette_scores
)


best_k = list(
    K_values
)[best_position]


best_silhouette = silhouette_scores[
    best_position
]


print("OPTIMAL K")


print(
    "\nBest value of k:",
    best_k
)


print(
    "Best Silhouette Score:",
    round(
        best_silhouette,
        4
    )
)

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    list(K_values),

    inertias,

    marker="o"

)


plt.xlabel(
    "Number of clusters (k)"
)


plt.ylabel(
    "Inertia"
)


plt.title(
    "Elbow Method"
)


plt.xticks(
    list(K_values)
)


plt.grid()


plt.tight_layout()


plt.show()


plt.figure(
    figsize=(8, 5)
)


plt.plot(

    list(K_values),

    silhouette_scores,

    marker="o"

)


plt.xlabel(
    "Number of clusters (k)"
)


plt.ylabel(
    "Silhouette Score"
)


plt.title(
    "Silhouette Method"
)


plt.xticks(
    list(K_values)
)


plt.grid()


plt.tight_layout()


plt.show()

final_model = KMeans(

    n_clusters=best_k,

    random_state=42,

    n_init=10

)


clusters = final_model.fit_predict(
    X_preprocessed
)

data["Cluster"] = clusters



print("DATASET WITH CLUSTERS")



print(
    data.head(20)
)


print("OBSERVATIONS PER CLUSTER")



cluster_counts = data[
    "Cluster"
].value_counts().sort_index()


print(
    cluster_counts
)


pca = PCA(
    n_components=2
)


X_pca = pca.fit_transform(
    X_preprocessed
)

plt.figure(
    figsize=(8, 6)
)


scatter = plt.scatter(

    X_pca[:, 0],

    X_pca[:, 1],

    c=clusters

)


plt.xlabel(
    "Principal Component 1"
)


plt.ylabel(
    "Principal Component 2"
)


plt.title(
    "K-Means Clusters"
)


plt.colorbar(
    scatter,
    label="Cluster"
)


plt.grid()


plt.tight_layout()


plt.show()

output_file = (
    r"C:\Documentos\2SEM2026\intelligence Artificial"
    r"\lab assigment 1\personas_desaparecidas_clusters.xlsx"
)


data.to_excel(

    output_file,

    index=False

)
