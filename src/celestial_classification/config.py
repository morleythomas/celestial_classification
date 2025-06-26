random_seed = 123

class dataset:
    dir = "../data/star_classification.csv" # Relative to notebooks folder
    test_size = .4

    extract_cols = [
        "redshift",
        "u",
        "g",
        "r",
        "i",
        "z",
        "class"
    ]

class training:
    model_list = ["rf", "logr", "gnb", "knn", "dt", "mlp", "xgb"]

    hyperparameters = {
        "rf": {
            "random_state": random_seed
        },
        "logr": {
            "random_state": random_seed,
            "max_iter": 1000
        },
        "gnb": {
            
        },
        "knn": {
            "n_jobs": -1
        },
        "dt": {
            "random_state": random_seed
        },
        "xgb": {
            "random_state": random_seed
        },
        "mlp": {
            "hidden_size": 32,
            "n_epochs": 5,
            "lr": 0.001,
            "batch_size": 16, 
            "shuffle_dataset": False
        }
    }

    kfold_parameters = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": random_seed
    }

class plots:
    rocauc_grid_cols = 3