"""
Central registry of ML models, DL architectures, and imbalance-handling
techniques. Adding a new model/technique later means adding one entry here
and nowhere else in the codebase.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Any


@dataclass
class ModelSpec:
    key: str
    label: str
    family: str  # "ML" or "DL"
    description: str
    builder: Callable[..., Any]  # (random_state, class_weight) -> estimator (ML)
                                  # or (input_dim, n_classes) -> keras model (DL)


# ---------------------------------------------------------------------------
# Machine Learning models
# ---------------------------------------------------------------------------

def _build_ml_models() -> Dict[str, ModelSpec]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
    )
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    specs: Dict[str, ModelSpec] = {}

    def add(key, label, description, builder):
        specs[key] = ModelSpec(key, label, "ML", description, builder)

    add("logistic_regression", "Logistic Regression",
        "Linear baseline classifier; fast and interpretable.",
        lambda rs, cw: LogisticRegression(max_iter=1000, random_state=rs,
                                           class_weight=cw))

    add("decision_tree", "Decision Tree",
        "Simple tree-based model, prone to overfitting but interpretable.",
        lambda rs, cw: DecisionTreeClassifier(random_state=rs, class_weight=cw))

    add("random_forest", "Random Forest",
        "Ensemble of decision trees; strong general-purpose baseline.",
        lambda rs, cw: RandomForestClassifier(
            n_estimators=200, random_state=rs, class_weight=cw, n_jobs=-1))

    add("svm", "Support Vector Machine",
        "Margin-based classifier, effective in high-dimensional spaces.",
        lambda rs, cw: SVC(probability=True, random_state=rs, class_weight=cw))

    add("knn", "K-Nearest Neighbors",
        "Instance-based learner; sensitive to feature scaling.",
        lambda rs, cw: KNeighborsClassifier(n_neighbors=5))

    add("naive_bayes", "Naive Bayes",
        "Probabilistic classifier assuming feature independence.",
        lambda rs, cw: GaussianNB())

    add("gradient_boosting", "Gradient Boosting",
        "Sequential ensemble of shallow trees; strong tabular performance.",
        lambda rs, cw: GradientBoostingClassifier(random_state=rs))

    add("adaboost", "AdaBoost",
        "Boosted ensemble that reweights misclassified samples.",
        lambda rs, cw: AdaBoostClassifier(random_state=rs))

    try:
        from xgboost import XGBClassifier

        add("xgboost", "XGBoost",
            "Gradient-boosted trees with regularization; industry standard.",
            lambda rs, cw: XGBClassifier(
                random_state=rs, eval_metric="logloss",
                use_label_encoder=False, n_jobs=-1))
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        add("lightgbm", "LightGBM",
            "Histogram-based gradient boosting; fast on large tabular data.",
            lambda rs, cw: LGBMClassifier(
                random_state=rs, class_weight=cw, verbosity=-1))
    except ImportError:
        pass

    return specs


# ---------------------------------------------------------------------------
# Deep Learning architectures
# All DL models consume tabular features. Sequence-style architectures
# (CNN/LSTM/etc.) treat the feature vector as a length-N sequence with a
# single channel, which is a standard way to apply them to tabular data.
# ---------------------------------------------------------------------------

def _build_dl_models() -> Dict[str, ModelSpec]:
    specs: Dict[str, ModelSpec] = {}

    def add(key, label, description, builder):
        specs[key] = ModelSpec(key, label, "DL", description, builder)

    def _compile(model, n_classes):
        import tensorflow as tf
        loss = "binary_crossentropy" if n_classes <= 2 else "sparse_categorical_crossentropy"
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss,
                       metrics=["accuracy"])
        return model

    def ann(input_dim, n_classes):
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential
        out_units, out_act = (1, "sigmoid") if n_classes <= 2 else (n_classes, "softmax")
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(out_units, activation=out_act),
        ])
        return _compile(model, n_classes)

    def mlp(input_dim, n_classes):
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential
        out_units, out_act = (1, "sigmoid") if n_classes <= 2 else (n_classes, "softmax")
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(out_units, activation=out_act),
        ])
        return _compile(model, n_classes)

    def _reshape_input():
        from tensorflow.keras import layers
        return layers.Reshape((-1, 1))

    def cnn(input_dim, n_classes):
        from tensorflow.keras import layers, Sequential
        out_units, out_act = (1, "sigmoid") if n_classes <= 2 else (n_classes, "softmax")
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            _reshape_input(),
            layers.Conv1D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation="relu"),
            layers.Dense(out_units, activation=out_act),
        ])
        return _compile(model, n_classes)

    def cnn_1d(input_dim, n_classes):
        # A slightly deeper 1D-CNN variant, kept distinct from `cnn`.
        from tensorflow.keras import layers, Sequential
        out_units, out_act = (1, "sigmoid") if n_classes <= 2 else (n_classes, "softmax")
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            _reshape_input(),
            layers.Conv1D(64, 5, activation="relu", padding="same"),
            layers.Conv1D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(out_units, activation=out_act),
        ])
        return _compile(model, n_classes)

    def lstm(input_dim, n_classes):
        from tensorflow.keras import layers, Sequential
        out_units, out_act = (1, "sigmoid") if n_classes <= 2 else (n_classes, "softmax")
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            _reshape_input(),
            layers.LSTM(64),
            layers.Dense(32, activation="relu"),
            layers.Dense(out_units, activation=out_act),
        ])
        return _compile(model, n_classes)

    def gru(input_dim, n_classes):
        from tensorflow.keras import layers, Sequential
        out_units, out_act = (1, "sigmoid") if n_classes <= 2 else (n_classes, "softmax")
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            _reshape_input(),
            layers.GRU(64),
            layers.Dense(32, activation="relu"),
            layers.Dense(out_units, activation=out_act),
        ])
        return _compile(model, n_classes)

    def bilstm(input_dim, n_classes):
        from tensorflow.keras import layers, Sequential
        out_units, out_act = (1, "sigmoid") if n_classes <= 2 else (n_classes, "softmax")
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            _reshape_input(),
            layers.Bidirectional(layers.LSTM(48)),
            layers.Dense(32, activation="relu"),
            layers.Dense(out_units, activation=out_act),
        ])
        return _compile(model, n_classes)

    def bigru(input_dim, n_classes):
        from tensorflow.keras import layers, Sequential
        out_units, out_act = (1, "sigmoid") if n_classes <= 2 else (n_classes, "softmax")
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            _reshape_input(),
            layers.Bidirectional(layers.GRU(48)),
            layers.Dense(32, activation="relu"),
            layers.Dense(out_units, activation=out_act),
        ])
        return _compile(model, n_classes)

    def cnn_lstm(input_dim, n_classes):
        from tensorflow.keras import layers, Sequential
        out_units, out_act = (1, "sigmoid") if n_classes <= 2 else (n_classes, "softmax")
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            _reshape_input(),
            layers.Conv1D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2),
            layers.LSTM(48),
            layers.Dense(32, activation="relu"),
            layers.Dense(out_units, activation=out_act),
        ])
        return _compile(model, n_classes)

    def attention_nn(input_dim, n_classes):
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        out_units, out_act = (1, "sigmoid") if n_classes <= 2 else (n_classes, "softmax")
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Reshape((-1, 1))(inputs)
        x = layers.Dense(32)(x)
        attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(out_units, activation=out_act)(x)
        model = Model(inputs, outputs)
        return _compile(model, n_classes)

    add("ann", "ANN", "Fully-connected feed-forward artificial neural network.", ann)
    add("mlp", "MLP", "Deeper multi-layer perceptron with batch normalization.", mlp)
    add("cnn", "CNN", "1D convolutional network treating features as a signal.", cnn)
    add("cnn_1d", "1D-CNN (Deep)", "Deeper 1D-CNN variant with stacked conv layers.", cnn_1d)
    add("lstm", "LSTM", "Long Short-Term Memory recurrent network.", lstm)
    add("gru", "GRU", "Gated Recurrent Unit network, a lighter alternative to LSTM.", gru)
    add("bilstm", "BiLSTM", "Bidirectional LSTM reading the feature sequence both ways.", bilstm)
    add("bigru", "BiGRU", "Bidirectional GRU network.", bigru)
    add("cnn_lstm", "CNN-LSTM", "Convolutional feature extractor feeding an LSTM.", cnn_lstm)
    add("attention_nn", "Attention Network", "Self-attention based classifier over feature tokens.", attention_nn)

    return specs


# ---------------------------------------------------------------------------
# Imbalance-handling techniques
# ---------------------------------------------------------------------------

@dataclass
class ImbalanceSpec:
    key: str
    label: str
    description: str
    kind: str  # "resample", "class_weight", "none"


IMBALANCE_METHODS: Dict[str, ImbalanceSpec] = {
    "none": ImbalanceSpec("none", "No Balancing",
                           "Use the training data as-is.", "none"),
    "random_oversample": ImbalanceSpec("random_oversample", "Random Oversampling",
                                        "Duplicate minority-class samples until classes are balanced.", "resample"),
    "random_undersample": ImbalanceSpec("random_undersample", "Random Undersampling",
                                         "Remove majority-class samples until classes are balanced.", "resample"),
    "smote": ImbalanceSpec("smote", "SMOTE",
                            "Synthesize new minority samples by interpolating between neighbors.", "resample"),
    "adasyn": ImbalanceSpec("adasyn", "ADASYN",
                             "Adaptive synthetic sampling focused on harder-to-learn minority samples.", "resample"),
    "smoteenn": ImbalanceSpec("smoteenn", "SMOTEENN",
                               "SMOTE oversampling combined with Edited Nearest Neighbours cleaning.", "resample"),
    "smotetomek": ImbalanceSpec("smotetomek", "SMOTETomek",
                                 "SMOTE oversampling combined with Tomek-link removal.", "resample"),
    "class_weight": ImbalanceSpec("class_weight", "Class Weighting",
                                   "Penalize misclassifying the minority class more heavily during training.", "class_weight"),
}


ML_MODELS: Dict[str, ModelSpec] = _build_ml_models()
DL_MODELS: Dict[str, ModelSpec] = _build_dl_models()
ALL_MODELS: Dict[str, ModelSpec] = {**ML_MODELS, **DL_MODELS}
