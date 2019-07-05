from components.classification import ClassifierChoice
from components.data_preprocessing import DataPreprocessorChoice

BINARY_CLASSIFICATION = 1
MULTICLASS_CLASSIFICATION = 2
MULTILABEL_CLASSIFICATION = 3
REGRESSION = 4

TASK_TYPES_TO_STRING = {
    BINARY_CLASSIFICATION: "binary.classification",
    MULTICLASS_CLASSIFICATION: "multiclass.classification",
    MULTILABEL_CLASSIFICATION: "multilabel.classification",
    REGRESSION: "regression"
}

STRING_TO_TASK_TYPES = {
    "binary.classification": BINARY_CLASSIFICATION,
    "multiclass.classification": MULTICLASS_CLASSIFICATION,
    "multilabel.classification": MULTILABEL_CLASSIFICATION,
    "regression": REGRESSION
}

TASK_TYPES_TO_CHOICE = {
    ClassifierChoice.TASK_NAME(): ClassifierChoice(),
    DataPreprocessorChoice.TASK_NAME(): DataPreprocessorChoice()
}
