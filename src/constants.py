"""General constants for constant stuff"""

ORDINALS = ["BsmtQual", "ExterQual", "ExterCond", "FireplaceQu", "KitchenQual"]
ORDINALS_ORDERING = ["Missing", "Po", "Fa", "TA", "Gd", "Ex"]
DROP_COLUMNS = ["GarageYrBlt", "YrSold", "Exterior2nd", "PoolQC"]

# default params
DEFAULT_REDUNDANT_COLUMNS_THRESHOLD = 0.9
DEFAULT_NAN_COLUMNS_THRESHOLD = 0.9
DEFAULT_RARE_CATEGORIES_DROP_THRESHOLD = 0.1

DEFAULT_PREPROCESS_INPUT_PATH = "data/raw/train.csv"
DEFAULT_PREPROCESS_OUTPUT_PATH = "data/interim/train.csv"

DEFAULT_X_TRAIN_PATH = "data/model_input/x_train.csv"
DEFAULT_X_TEST_PATH = "data/model_input/x_test.csv"
DEFAULT_Y_TRAIN_PATH = "data/model_input/y_train.csv"
DEFAULT_Y_TEST_PATH = "data/model_input/y_test.csv"

# arguments helps
INPUT_PATH_HELP = "The path of the csv file to preprocess"
OUTPUT_PATH_HELP = "Where to store the processed path"
REDUNDANT_COLUMNS_HELP = "A threshold to use above which to drop drop nan columns"
NAN_COLUMNS_THRESHOLD_HELP = "Drop if the column is mostly the same"
RARE_CATEGORIES_DROP_THRESHOLD_HELP = "Replace rare categories"
VERBOSE_HELP = "Whether to print the steps of the pipeline"

CATEGORICAL_COLUMNS = [
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "BsmtCond",
    "Alley",
    "Fence",
    "RoofStyle",
    "MiscFeature",
    "Heating",
    "CentralAir",
    "Electrical",
    # 'BsmtHalfBath',
    # 'KitchenAbvGr',
    "Functional",
    "PavedDrive",
    "Street",
    "Utilities",
    "LandSlope",
    "Condition1",
    "Condition2",
    "RoofMatl",
    "MSZoning",
    "LotShape",
    "LandContour",
    "LotConfig",
    "Neighborhood",
    "Condition1",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "Exterior1st",
    "Exterior2nd",
    "Foundation",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "HeatingQC",
    "GarageType",
    "GarageFinish",
    "SaleType",
    "SaleCondition",
    # "BsmtQual", "ExterQual", "ExterCond", "FireplaceQu", "KitchenQual",
    "MasVnrType",
]

NUMERICAL_COLUMNS = [
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "BsmtFullBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "MiscVal",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "GarageCars",
    "MoSold",
    "LotFrontage",
    "LotArea",
    "MSSubClass",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "MoSold",
    "MasVnrArea",
    "BsmtFinSF1",
    "LowQualFinSF",
    "3SsnPorch",
    "ScreenPorch",
]

# training
DEFAULT_TRAIN_INPUT_PATH = "data/interim/train.csv"
DEFAULT_MODEL_PATH = "data/models/model.pickle"


# default params
DEFAULT_REDUNDANT_COLUMNS_THRESHOLD = 0.9
DEFAULT_NAN_COLUMNS_THRESHOLD = 0.9
DEFAULT_RARE_CATEGORIES_DROP_THRESHOLD = 0.1

DEFAULT_PREPROCESS_INPUT_PATH = "data/raw/train.csv"
DEFAULT_PREPROCESS_OUTPUT_PATH = "data/interim/train.csv"

DEFAULT_TRAIN_INPUT_PATH = "data/interim/train.csv"

# arguments helps
INPUT_PATH_HELP = "The path of the csv file to preprocess"
OUTPUT_PATH_HELP = "Where to store the processed path"
REDUNDANT_COLUMNS_HELP = "A threshold to use above which to drop drop nan columns"
NAN_COLUMNS_THRESHOLD_HELP = "Drop if the column is mostly the same"
RARE_CATEGORIES_DROP_THRESHOLD_HELP = "Replace rare categories"


DEFAULT_SPLIT_SEED = 1
DEFAULT_TRAIN_SET_SIZE = 0.8
DEFAULT_LASSO_ALPHA = 0.05
DEFAULT_MODEL_SEED = 1

TARGET_VARIABLE_NAME = "SalePrice"
ID_COLUMN = "Id"