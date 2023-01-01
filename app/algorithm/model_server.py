import numpy as np, pandas as pd
import os, sys
import json
from shap import Explainer

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.classifier as classifier


# get model configuration parameters
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema):
        self.model_path = model_path
        self.data_schema = data_schema
        self.preprocessor = None
        self.model = None
        self.id_field_name = self.data_schema["inputDatasets"][
            "binaryClassificationBaseMainInput"
        ]["idField"]
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 5

    def _get_preprocessor(self):
        if self.preprocessor is None:
            self.preprocessor = pipeline.load_preprocessor(self.model_path)
        return self.preprocessor

    def _get_model(self):
        if self.model is None:
            self.model = classifier.load_model(self.model_path)
        return self.model

    def _get_predictions(self, data):
        preprocessor = self._get_preprocessor()
        model = self._get_model()

        if preprocessor is None:
            raise Exception("No preprocessor found. Did you train first?")
        if model is None:
            raise Exception("No model found. Did you train first?")

        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)
        # Grab input features for prediction
        pred_X, ids = proc_data["X"].astype(np.float), proc_data["ids"]
        # make predictions
        preds = model.predict(pred_X)
        return preds, ids

    def predict_proba(self, data):
        preds, pred_ids = self._get_predictions(data)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        id_df = pd.DataFrame(pred_ids, columns=[self.id_field_name])

        # return the prediction df with the id and class probability fields
        preds_df = pd.concat([id_df, pd.DataFrame(preds, columns=class_names)], axis=1)
        return preds_df

    def predict_to_json(self, data):
        print("data", data.shape)
        preds_df = self.predict_proba(data)
        print("preds_df", preds_df.shape)
        class_names = preds_df.columns[1:]
        preds_df["__label"] = pd.DataFrame(
            preds_df[class_names], columns=class_names
        ).idxmax(axis=1)

        predictions_response = []
        for rec in preds_df.to_dict(orient="records"):
            pred_obj = {}
            pred_obj[self.id_field_name] = rec[self.id_field_name]
            pred_obj["label"] = rec["__label"]
            pred_obj["probabilities"] = {
                str(k): np.round(v, 5)
                for k, v in rec.items()
                if k not in [self.id_field_name, "__label"]
            }
            predictions_response.append(pred_obj)
        print("predictions_response", predictions_response)
        return predictions_response

    def predict(self, data):
        preds_df = self.predict_proba(data)
        class_names = [str(c) for c in preds_df.columns[1:]]
        preds_df["prediction"] = pd.DataFrame(
            preds_df[class_names], columns=class_names
        ).idxmax(axis=1)
        preds_df.drop(class_names, axis=1, inplace=True)
        return preds_df

    def _get_target_class_proba(self, X):
        """
        Returns predicted probability of the target class
        """
        model = self._get_model()
        preds = model.predict(X)
        return preds[:, 1]

    def explain_local(self, data):

        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f"""Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations."""
            print(msg)

        preprocessor = self._get_preprocessor()
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data.head(self.MAX_LOCAL_EXPLANATIONS))
        # ------------------------------------------------------------------------------
        # original class predictions

        model = self._get_model()
        pred_X = proc_data["X"].astype(np.float)
        X_cols = proc_data["X_cols"]
        ids = proc_data["ids"]

        pred_target_class_prob = model.predict(pred_X)[:, 1]

        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        # ------------------------------------------------------------------------------
        print(f"Generating local explanations for {pred_X.shape[0]} sample(s).")
        # create the shapley explainer
        mask = np.zeros_like(pred_X)
        explainer = Explainer(self._get_target_class_proba, mask, seed=1)
        # Get local explanations
        shap_values = explainer(pd.DataFrame(pred_X, columns=X_cols))

        # ------------------------------------------------------------------------------
        # create json objects of explanation scores
        N = pred_X.shape[0]
        explanations = []
        for i in range(N):            

            target_class_prob = pred_target_class_prob[i]
            pred_class = class_names[0] if target_class_prob < 0.5 else class_names[1]
            pred_class_prob = (
                target_class_prob if target_class_prob > 0.5 else 1 - target_class_prob
            )

            other_class = str(
                class_names[0] if class_names[1] == pred_class else class_names[1]
            )
            probabilities = {
                other_class: np.round(1 - pred_class_prob, 5),
                pred_class: np.round(pred_class_prob, 5),
            }

            sample_expl_dict = {}
            sample_expl_dict["baseline"] = np.round(shap_values.base_values[i], 5)

            feature_scores = {}
            for f_num, feature in enumerate(shap_values.feature_names):
                feature_scores[feature] = round(shap_values.values[i][f_num], 5)

            sample_expl_dict["feature_scores"] = feature_scores

            explanations.append({
                self.id_field_name: ids[i],
                "label": str(pred_class),
                "probabilities": probabilities,
                "explanations": sample_expl_dict,
            })

        # ------------------------------------------------------
        """
        To plot the shapley values:
        you can only plot one sample at a time. 
        if you want to plot all samples. create a loop and use the index (sample_idx)
        """
        # sample_idx = 4
        # shap_values.base_values = shap_values.base_values[sample_idx]
        # shap_values.values = shap_values.values[sample_idx]
        # shap_values.data = shap_values.data[sample_idx]
        # shap.plots.waterfall(shap_values)
        # ------------------------------------------------------
        explanations = {"predictions": explanations}
        explanations = json.dumps(explanations, cls=utils.NpEncoder, indent=2)
        return explanations
